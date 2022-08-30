# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# Copyright 2018-2022, Ben Cardoen
using Test
using SPECHT
using Random
using Statistics
using Logging
import Images
import ERGO
import LinearAlgebra
using ImageFiltering


@testset "SPECHT.jl" begin

	@testset "hm" begin
		@test harmonicmean([1, 2, 3]) > 0
	end


	@testset "insilico" begin
		g, t, r, r1 = generate_scenario(256, 256, 20, 20)
		g2, t2, r2, r3 = generate_scenario(256, 256, 40, 40)
		sum(g) < sum(g2)
	end

	@testset "noise" begin
		g=gaussiannoise(zeros(10, 10), 1; μ=0, bits=8)
		k=gaussiannoise(zeros(10, 10), 4; μ=0, bits=8)
		@test sum(g) < sum(k)
		p=poissonnoise(zeros(10,10), 1; bits=8)
		p2=poissonnoise(zeros(10,10), 10; bits=8)
		@test sum(p) < sum(p2)
		Random.seed!(42)
		p3=sandp(1, zeros(10,10))[2]
		p4=sandp(10, zeros(10,10))[2]
		@test sum(p3) > sum(p4)
	end

	@testset "apo" begin
		Q = zeros(100, 100)
		Q[20:20, 20:20] .= 1
		ccs=Images.label_components(Q)
		r = annotate_spots(Q, ccs, [.42])
		@test Images.N0f16.(0.42) ∈ unique(r)
	end

	@testset "fth" begin
		Q=ones(10, 10)
		Q[2:2, 2:2] .= 0
		r=filterth(Q, .5)
		@test length(r) == 1
	end

	@testset "t3" begin
		r=tcolors([zeros(10, 10)], x->x)
		@test eltype(r) <: Images.RGB
		r=tcolors(zeros(10, 10), zeros(10, 10), zeros(10, 10), x->x)
		@test eltype(r) <: Images.RGB
	end

	@testset "cd" begin
		d="Q"
		if isdir(d)
			rm(d)
		end
		@test ! isdir(d)
		checkdir(d)
		@test isdir(d)
		rm(d)
	end


	@testset "ots" begin

		Q = Random.rand(100, 100)
		Q[Q .< .5] .= 0.01
		a, b = computeotsu(Q)

		@test sum(b .* Q) < sum(a .* Q) < sum(Q)
		_ = computelogotsu(Q)
	end

	@testset "pcell" begin
		Random.seed!(42)
		A = zeros(256, 256)
		A[20:30, 40:60] .= max.(0.1 .+ rand(11, 21), 1)
		slices = [A for _ in 1:3]
		r=process_non_segmented_non_decon_channels(slices; PRC=1, sigma=1, quantile=0.9, pixprec=3)
		@test r[1] == r[2]
		@test 232<sum(r[1][1])<233
	end

	@testset "vismask" begin
		res = Dict()
		res[1] = [zeros(10, 10)]
		res[1][end][5:6, 5:6] .= 1
		res[2] = [zeros(10, 10)]
		res[2][end][5:6, 5:6] .= 1
		res[0] = [zeros(10, 10)]
		res[0][end][5:6, 5:6] .= 1
		a, b = computemasks(res, res[0][end])
		@test sum(a) == sum(b)
	end

	@testset "cv" begin
		@test cycle_vec_1([1,2,3]) == [2,3,1]
	end

	@testset "cs" begin
		@test csum(1, 2) == 1
	end

	@testset "scoring" begin
		Random.seed!(42)
		R = Random.rand(100, 100)
		Q = copy(R)
		R[R .< 0.5] .= 0
		Q[Q .< 0.25] .= 0
		fp, tp, fn = score_masks_visual(ERGO.tomask(Q), ERGO.tomask(R))
		fp2, tp2, fn2= score_masks_visual(ERGO.tomask(Q), ERGO.tomask(Q))
		@test sum(tp) < sum(tp2)
		Random.seed!(42)
		R = Random.rand(100, 100)
		Q = copy(R)
		R[R .< 0.5] .= 0
		Q[Q .< 0.45] .= 0
		Q[Q .> 0.75] .= 0
		fp, tp, fn, m = score_masks_mcc(ERGO.tomask(R), ERGO.tomask(Q))
		abs(m)-abs(0.249702443995) < 0.01
	end

	@testset "CS" begin
		Random.seed!(42)
		for _ in 1:100
			A = rand(20,20)
			B = zeros(40,40)
			B[10:29, 10:29] .= A
			d = cellstats(B)
			@test 0 <= d[:anisotropy] <= 1
			@test d[:λ1] >= d[:λ2]
			@test d[:area] == 400
		end
	end

	@testset "contrastive" begin

		Random.seed!(2)
		for _ in 1:100
			N = 100
			from = rand(N, 5)
			to = rand(N+5, 5)
			to2 = 0.5 .+ rand(N+5, 5)
			px1 = contrast_x_to_y(from, to)
			px2 = contrast_x_to_y(from, to2)
			@test all(0 .<= px1 .<= 1)
			@test all(0 .<= px2 .<= 1)
			@test Statistics.mean(px1) > Statistics.mean(px2)
		end
	end

    @testset "qmito" begin
        C1C2 = zeros(20, 20)
        C1C2[3:3,9:15] .= 1
        C1C2[9:15,9:15] .= 1
        mito = zeros(20, 20)
        mito[16:20, 17:20] = rand(5, 4)
        mito[16:20, 1:5] = rand(5, 5)
        res, resmask = quantify_adjacent_mitophagy(C1C2, ERGO.tomask(mito), mito, 5, 10)
        @test iszero(res[1,1:end-3])
        @test ! iszero(res[2,:])
        @test res[2,1] == 2
		@test 0 < sum(resmask) <= sum(mito .* 1.01)
        isapprox(res[2, 2:3], [0.58234,  0.261352], atol=0.001)
		res, resmask = quantify_adjacent_mitophagy(C1C2, ERGO.tomask(mito), mito, 0, 1)
		# nspots, μ, σ, nz, μi, σi
		@test iszero(res[1:1:4])
		@test 0 <= sum(resmask) < sum(mito)
        C1C2 = zeros(20, 20)
        res, resmask = quantify_adjacent_mitophagy(C1C2, ERGO.tomask(mito), mito, 0, 1)
        @test isnothing(res)
		@test iszero(resmask)
    end

	@testset "aniso" begin
		Q = zeros(10, 10)
		Q[5:5, 5:5] .= 1
		a, l, m = aniso2(Q)
		@test a == 0 == l == m
		Random.seed!(52)
		for _ in 1:100
			Q = zeros(10, 10)
			Q[5:6,6:9] .= rand(2, 4)
			a, l, m = aniso2(Q)
			@test 0 < a < 1
			@test l > m
		end
	end


	@testset "iter" begin
        X = Random.rand(10, 10)
		Y = iterative(X)
		@test all(X .== Y)
		f = x -> x.^2
		Y = iterative(X, f, 1)
		@test all(f(X) == Y)
		Z = iterative(X, f, 3)
		@test Z != Y
		X = Random.rand(10, 10)
		Xp = iterative(X, x->x.^2, 0)
		@test all(Xp .== X)
    end

	@testset "qm12over" begin
		c12 = zeros(10, 10)
		mito = zeros(10, 10)
		c12[3:4,3:4] .= 1
		c12[6:7,6:8] .= 1
		mito[2:4, 2:4] .= rand(3,3)
		res = quantify_c12_mito_overlap(c12, mito, mito)
		@test isapprox(res[1, 2], Statistics.mean(mito[3:4, 3:4]), atol=0.001)
		@test isapprox(res[1, 3], Statistics.std(mito[3:4, 3:4]), atol=0.001)
		@test res[1,1] == 4
	end

    @testset "checkdrift" begin
        A = zeros(10,10)
		A[1,1] = 1
		@test checkdrift(A, 1) == true
		A = zeros(10,10)
		A[2,2] = 1
		@test checkdrift(A, 2) == true
    end

	@testset "checkaligned" begin
		A=zeros(Images.N0f8, 5,5)
		B=zeros(Images.N0f8, 5,5)
		C=zeros(Images.N0f8, 5,5)
		A[1,1] = 1
		B[2,2] = 1
		C[2,2] = 1
		@test checksegmentsaligned([A,B,C]) == false
		A[2,2] = 1
		@test checksegmentsaligned([A,B,C]) == true
	end

	@testset "ir" begin
		for _ in 1:10
        	X = Random.rand(2)
			Y = id(X)
			@test X==Y
			X[1] = 7
			@test X!=Y
		end
    end

	@testset "quantifyadjc12" begin
		c1c2_img = zeros(10,10)
		mito_img = zeros(10,10)
		md, mm = quantify_adjacent_c12(mito_img, c1c2_img)
		@test isnothing(md)
		@test iszero(mm)

		c1c2_img[4:6, 4:6] .= 1
		mito_img[5:10, 5:10] .= 1
		md, mm = quantify_adjacent_c12(mito_img, c1c2_img)
		@test iszero(md)
		@test sum(mm) == 9

		c1c2_img = zeros(10,10)
		mito_img = zeros(10,10)
		mito_img[5:10, 5:10] .= 1
		md, mm = quantify_adjacent_c12(mito_img, c1c2_img)
		@test all(isinf.(md))
		@test iszero(mm)

		c1c2_img = zeros(10,10)
		mito_img = zeros(10,10)
		c1c2_img[1:2, 1:2] .= 1
		mito_img[5:10, 5:10] .= 1
		md, mm = quantify_adjacent_c12(mito_img, c1c2_img)
		@test all(md .> 0)
		@test sum(mm) == 4

		c1c2_img = zeros(10,10)
		mito_img = zeros(10,10)
		c1c2_img[1:2, 1:2] .= 1
		c1c2_img[9:10, 9:10] .= 1
		mito_img[5:8, 5:8] .= 1
		md, mm = quantify_adjacent_c12(mito_img, c1c2_img)
		@test length(md) == 1
		@test sum(mm) == 4
		@test isapprox(md[1], sqrt(2), atol=0.1)
	end


    @testset "cohend" begin
		Random.seed!(42)
		for i in 1:1000
			A = rand(100)
			B = rand(100) .+ 1
			d = cohen_d(A, A)
			@test iszero(d)
			dab = cohen_d(A, B)
			dba = cohen_d(B, A)
			@test isapprox(dab, -dba)
			@test ! iszero(dba)
		end
	end

    @testset "qmito_survive_zero" begin
        C1C2 = zeros(20, 20)
		C1C3 = zeros(20, 20)
		C1C4 = zeros(20, 20)
		res = Dict()
		for i in 0:2
			res[i] = Images.label_components(C1C2), ERGO.tomask(C1C2)
		end
		res[3] = Dict(0=>C1C2, 1=>C1C2, 2=>C1C2)
		results = qspot(res, true)
		results = qspot(res, false)
    end

	@testset "qmitoc12" begin
		m0 = zeros(100,100)
		m1 = zeros(100,100)
		m0[50:52,50:52] = rand(3,3)
		m1[10:20,10:20] .= rand(11, 11)
		m1[40:45,40:45] .= rand(6, 6)
		res, img = quantify_nearest_mitophagy(ERGO.tomask(m0), ERGO.tomask(m1), m1)
		@test !isnothing(res)
		@test !iszero(img)
		@test res[1,1] < 10
		@test isapprox(res[1,2],  Statistics.mean(m1[40:45,40:45]), atol=0.1)
		@test res[1,3] == 36
		m0 = zeros(100,100)
		m1 = zeros(100,100)
		res, img = quantify_nearest_mitophagy(ERGO.tomask(m0), ERGO.tomask(m1), m1)
		@test isnothing(res)
		@test iszero(img)
		m0 = zeros(100,100)
		m0[50, 50] = 1
		m1 = zeros(100,100)
		res, img = quantify_nearest_mitophagy(ERGO.tomask(m0), ERGO.tomask(m1), m1)
		@test isinf(res[1, 1])
		@test iszero(res[1, 2:3])
		@test iszero(img)
	end

	@testset "qmito_case_1" begin
        C1 = zeros(20, 20)
		C2 = zeros(20, 20)
		C1[8:13,8:13] .= 1
		C2[10:15,10:15] .= 1
		mito = zeros(20, 20)
		mito[16:20, 17:20] = rand(5, 4)
		mito[16:20, 1:5] = rand(5, 5)
		res = Dict()
		res[0] = Images.label_components(mito), ERGO.tomask(mito)
		res[1] = Images.label_components(C1), ERGO.tomask(C1)
		res[2] = Images.label_components(C2), ERGO.tomask(C2)
		res[3] = Dict(0=>mito, 1=>C1, 2=>C1)
		results = qspot(res, true)
		@test results[1:6] == (1, 1, 1, 0, 0, 0)
		@test results[7][1][1] == sum(C1)
		@test results[7][2][1] == sum(C2)
		@test results[7][3][1] == sum(ERGO.tomask(C1 .* C2))
		results = qspot(res, false)
		@test results[1:6] == (1, 1, 1, 0, 0, 0)
		@test results[7][1][1] == sum(C1)
		@test results[7][2][1] == sum(C2)
		@test results[7][3][1] == sum(ERGO.tomask(C1 .+ C2))
		@test results[end-2] == ERGO.tomask(C1 .+ C2)
		@test isapprox(results[end-3][1], 2.23606797749979, atol=0.00001)
    end

	@testset "newqmito" begin
		Random.seed!(42)
		for _ in 1:100
			m0 = zeros(10,10)
			m1 = zeros(10,10)
			m2 = zeros(10,10)
			m0[5,5] = rand()
			m1[5:6,5] .= rand(2,)
			m2[3:4,2:5] .= rand(2, 4)
			m2[1:1,1:1] .= rand()
			res = Dict()
			res[0] = Images.label_components(ERGO.tomask(m0)), ERGO.tomask(m0)
			res[1] = Images.label_components(ERGO.tomask(m1)), ERGO.tomask(m1)
			res[2] = Images.label_components(ERGO.tomask(m2)), ERGO.tomask(m2)
			imgs = [m0, m1, m2]
			res[3] = Dict(0=>m0, 1=>m1, 2=>m2)
			c1count, c2count, c1c2count, c1c2mitocount, count_overlap_c1_mito, count_overlap_c2_mito, areas, distances, _fum, intensities, channelints = qspot(res, false)
		end
		# intensities
		# channelints
	end

	@testset "fth" begin
        img = zeros(100, 100)
		img[45:55, 45:55] .= 1
		sigmas = [3, 3]
		em = copy(img)
		em[44:56, 44:56] .=1
		em = em .- img
		smoothedmedian = imfilter(img, Kernel.gaussian((sigmas)))
		for z in 0.5:0.125:3.5
			imgl = ImageFiltering.imfilter(smoothedmedian, ImageFiltering.Kernel.Laplacian());
			th, _ = find_th(imgl, z, false, 1, nothing)
			@test th > 0
			th, _ = find_th(imgl, z, false, 1, em)
			@test th > 0
			th, _ = find_th(imgl, z, true, 1, em)
			@test th > 0
		end
    end

	@testset "snrsegment" begin
		img = zeros(100, 100)
		img[45:55, 45:55] .= 1
		for k in 1:200
			x = Int(round(rand() * 99)) + 1
			y = Int(round(rand() * 99)) + 1
			img[x,y] = rand()
		end
		lastsum = 0
		for i in 1:5
			i,j,k = SPECHT.lowsnrsegment(ERGO.normimg(img), edgeiters=i)
			@test sum(k) > lastsum
			lastsum = sum(k)
		end
	end

	@testset "qmito2" begin
		# function quantify_adjacent_mitophagy(c1c2_img, mito_img, raw_mito_img, TH_1, TH_2)
		c1c2_img = zeros(10,10)
		mito_img = zeros(10,10)
		raw_mito_img = zeros(10,10)
		TH_1 = 2
		TH_2 = 3
		res, rm = quantify_adjacent_mitophagy(c1c2_img, mito_img, raw_mito_img, TH_1, TH_2)
		@test isnothing(res)
		@test iszero(rm)
		c1c2_img[4:6, 4:6] .= 1
		mito_img[5:10, 5:10] .= 1
		raw_mito_img[5:10, 5:10] .= 0.5
		res, rm = quantify_adjacent_mitophagy(c1c2_img, mito_img, raw_mito_img, TH_1, TH_2)
		@test res[1,4] == sum(rm)/res[1,2]
		TH_2 = 2
		res2, rm2 = quantify_adjacent_mitophagy(c1c2_img, mito_img, raw_mito_img, TH_1, TH_2)
		@test res2[1,4] == sum(rm2)/res2[1,2]
		@test res2[1,4] < res[1,4]
		@test all(res[1:end-3] == res2[1:end-3])
		@test isapprox(res[1,5], Statistics.mean(raw_mito_img[raw_mito_img .> 0]))
		@test isapprox(res[1,6], Statistics.std(raw_mito_img[raw_mito_img .> 0]))
	end


    @testset "validatechannels" begin
		## Normal case
		tmp = mktempdir()
		C0, C1, C2 = [zeros(100, 100) for i in 1:3]
		images = C0, C1, C2
		for (i,img) in enumerate(images)
			Images.save(joinpath(tmp, "c$(i-1).tif"), img)
		end
		res = validatechannels(tmp)
		@test length(res) == 3
		rm(tmp, recursive=true)

		tmp = mktempdir()
		C0, C1, C2 = [zeros(100, 100) for i in 1:3]
		images = C0, C1, C2
		for (i,img) in enumerate(images)
			Images.save(joinpath(tmp, "c$(i-1).tif"), SPECHT.tcolors([img,img,img]))
		end
		res = validatechannels(tmp)
		@test isnothing(res)
		rm(tmp, recursive=true)

		## Missing mito, should work
		tmp = mktempdir()
		C1, C2 = [zeros(100, 100) for i in 1:2]
		images = C1, C2
		for (i,img) in enumerate(images)
			Images.save(joinpath(tmp, "c$(i).tif"), img)
		end
		res = validatechannels(tmp)
		@test length(res) == 3
		rm(tmp, recursive=true)

		### Not enough channels, should fail gracefully
		tmp = mktempdir()
		C1 = zeros(100, 100)
		images = [C1]
		for (i,img) in enumerate(images)
			Images.save(joinpath(tmp, "c$(i).tif"), img)
		end
		res = validatechannels(tmp)
		@test isnothing(res)
		rm(tmp, recursive=true)

		### Incorrect dimensions, should fail gracefully
		tmp = mktempdir()
		C1, C2 = zeros(100, 100), zeros(100, 101)
		images = [C1, C2]
		for (i,img) in enumerate(images)
			Images.save(joinpath(tmp, "c$(i).tif"), img)
		end
		res = validatechannels(tmp)
		@test isnothing(res)
		rm(tmp, recursive=true)
	end

	@testset "scale" begin
		Random.seed!(42)
		xs = rand(100)
		scaled = scale(xs)
		@test scaled == 0.0
		import Distributions
		Random.seed!(42)
		n = Distributions.Normal()
		xs = abs.(rand(n, 100))
		xs[90:100] .*= 5
		scaled = scale(xs)
		xs[90:100] .*= 5
		scaled1 = scale(xs)
		@test scaled1 > scaled
	end

	@testset "process_cell" begin
		# function process_cell(qpath, channels, outdir, serie, subdir, ct, experiment, z, selfscale, celln, SQR; maxspotsize=Inf64, segment=false, sigmas=[3,3])

		## Normal case
		tmp = mktempdir()
		out = mktempdir()
		C0, C1, C2 = [zeros(100, 100) for i in 1:3]
		images = C0, C1, C2
		for (i,img) in enumerate(images)
			Images.save(joinpath(tmp, "ch0$(i-1).tif"), img)
		end
		res = process_cell(tmp, [0,1,2], out, "na",  "na", "na", 1.5, false, 2, 5)
		rm(tmp, recursive=true)
		rm(out, recursive=true)

		## Segment
		Random.seed!(42)
		for i in 1:10
			tmp = mktempdir()
			out = mktempdir()
			img = zeros(100, 100)
			img[45:55, 45:55] .= 1
			for k in 1:200
				x = Int(round(rand() * 99)) + 1
				y = Int(round(rand() * 99)) + 1
				img[x,y] = rand()
			end
			C0, C1, C2 = [copy(img) for i in 1:3]
			images = C0, C1, C2
			for (i,img) in enumerate(images)
				Images.save(joinpath(tmp, "ch0$(i-1).tif"), img)
			end
			res = process_cell(tmp, [0,1,2], out, "na",  "na", "na", 1.5, false, 2, 5, mode="segment")
			@test length(res[1]) == 3
			@test length(res[end]) == 3
			@test res[3] == false
			rm(tmp, recursive=true)
			rm(out, recursive=true)
		end


		## Segment with 0, check if it does not crash
		tmp = mktempdir()
		out = mktempdir()
		C0, C1, C2 = [zeros(100, 100) for i in 1:3]
		images = C0, C1, C2
		for (i,img) in enumerate(images)
			Images.save(joinpath(tmp, "ch0$(i-1).tif"), img)
		end
		res = process_cell(tmp, [0,1,2], out, "na", "na",  "na", 1.5, false, 2, 5, mode="segment")
		@test length(res[1]) == 3
		@test res[3] == false
		rm(tmp, recursive=true)
		rm(out, recursive=true)

		tmp = mktempdir()
		out = mktempdir()
		C0, C1, C2 = [zeros(100, 100) for i in 1:3]
		images = C0, C1, C2
		for (i,img) in enumerate(images)
			Images.save(joinpath(tmp, "ch0$(i-1).tif"), img)
		end
		res = process_cell(tmp, [0,1,2], out, "na", "na",  "na", 1.5, false, 2, 5, mode="apery")
		rm(tmp, recursive=true)
		rm(out, recursive=true)
	end

	@testset "iqrf" begin
		Random.seed!(42)
		img = rand(1024, 1024)
		img[img .< 0.75] .= 0
		msk = ERGO.tomask(img)
		rf = filtercomponents_using_maxima_IQR(img, msk)
		@test sum(rf) == 216825
	end


	@testset "fth" begin
		Random.seed!(42)
		for i in 1:100
	        img = zeros(100, 100)
			img[45:55, 45:55] .= 1
			sigmas = [3, 3]
			em = copy(img)
			em[44:56, 44:56] .=1
			em = em .- img
			img = img + rand(100, 100)./5
			for z in 0.5:0.125:2.0
				res = process_tiffimage(img, z, sigmas, false, 1, 0, edgemask=em)
				@test maximum(res[1]) >= 1
				res = process_tiffimage(img, z, sigmas, false, 1, 0, edgemask=nothing)
				@test maximum(res[1]) >= 1
			end
		end
    end

    @testset "fk" begin
        Random.seed!(42)
        for _ in 1:10
            img = rand(100, 100, 10)
            u, s = Statistics.mean(img), Statistics.std(img)
            for k in 1.:4.
                ik, th = filter_k(img, k)
                @test all(ik[ik .> 0] .>= th)
                @test isapprox(u+k*s, th)
            end
            img = rand(100)
            img[25:45] .= 0.0
            ik, th = filter_k(img, 1, false)
            ikz, thz = filter_k(img, 1, true)
            @test thz > th
            @test sum(ik) > sum(ikz)
            img = 1.0 .+ rand(100)
            ik, th = filter_k(img, 1, false)
            ikz, thz = filter_k(img, 1, true)
            @test isapprox(thz, th, atol=0.02)
        end
        Random.seed!(42)
		n = Distributions.Normal()
        img = abs.(0.5 .+ rand(n, 100, 100))
		img[20:30] .*= 2
		img[40:60] .*= 4
        ikz, thz1 = filter_k(img, 1, true, true, 1.0)
		ikz, thz2 = filter_k(img, 1, true, true, 2.0)
		@test thz2 < thz1
    end

    @testset "hm" begin
        Random.seed!(42)
        for i in 1:100
            a = rand(100, 100)
            b = copy(a)
            h = harmonicmean(a)
            g = ERGO.gm(a)
            av = mean(a)
            @assert h <= g <= av
        end
    end

    @testset "stats" begin
        b = [0 0 2 3]
        a = [0 1 2 3]
        d, j = dice_jaccard(a, b)
        df, jf = dice_jaccard(a, a)
        @test df == 1.0
        @test isapprox(jf, 1.0)
        @test d == .8
        @test isapprox(j, 2/3.0)
    end

    @testset "filterccgeq" begin
        A = zeros(50, 50)
        A[1:5, 1:5] .= rand(5,5)
        A[11:14, 11:14] .= rand(4,4)
        A[21:24, 10:16] .= rand(4,7)
        Acc = Images.label_components(ERGO.tomask(A),trues(3,3))
        bmsk = SPECHT.filter_cc_sqr_greater_than(Acc, A, 4)
        rescc = Images.label_components(bmsk,trues(3,3))
        @test maximum(rescc) == 1
        bmsk = SPECHT.filter_cc_sqr_greater_than(Acc, A, 5)
        rescc = Images.label_components(bmsk,trues(3,3))
        @test maximum(rescc) == 0
    end

    @testset "IOU" begin
        A = zeros(4, 4)
        B = zeros(4, 4)
        AB = intersectimg(A, B)
        @test sum(AB) == 0.0
        AB = unionimg(A, B)
        @test sum(AB) == 0.0
        A[1,1] = 1
        A[1,2] = 1
        B[1,2] = 1
        AB = intersectimg(A, B)
        @test sum(AB) == 1
        AB = unionimg(A, B)
        @test sum(AB) == 2
    end

    @testset "rlap" begin
        Random.seed!(42)
        for _ in 1:1000
            b = 0.5 .- rand(100, 200, 4)
            rb = rlap(b)
            @test all(rb .>= 0)
        end
		bs = [-4, 2.0, 0]
		as = rlap(bs)
		@test sum(as) == 4
		qs = rlap!(bs)
		@test sum(as) == sum(qs)
		@test sum(as) == sum(bs)
    end

    @testset "mahalanobis" begin
        a = zeros(2,2)
        a[1,1] = 1.0
        a[2,2] = 1.0
        Random.seed!(42)
        for _ in 1:100
            b = rand(10, 2)
            b2 = rand(10, 2)
            euc = sqrt(sum((b2[1,:].-b[1,:]).^2))
            m = Statistics.mean(b, dims=1)
            @test iszero(mahalanobis(a, a, LinearAlgebra.inv(a)))
            an = Statistics.cov(b)
            for i in size(b2, 1)
                row = reshape(b2[i, :], 1, :)
                euc = sqrt(sum( (row .- m).^2))
                @test isapprox(mahalanobis(row, m, LinearAlgebra.inv(a)), euc)
                @test ! isapprox(mahalanobis(row, m, LinearAlgebra.inv(an)), euc)
            end
        end
    end

    @testset "cnl" begin
        Random.seed!(42)
        for _ in 1:100
            s1 = Random.rand(100, 100)
            s2 = Random.rand(100, 100)
            l1, l2, n1, n2 = compute_nl(s1, s2)
            @test all(n1 .>= 0)
            @test all(n2 .>= 0)
            @test (sum(n1) != 0)
            @test (sum(n2) != 0)
        end
    end

    @testset "ibound" begin
        left = zeros(10,10)
        left[4:8,3:5] .= 0.01 .+ rand()
        right = zeros(10,10)
        right[3:5,5:7] .= .5
        right[9:9,2:3] .= .3
        res, dp = computeintensityboundary(left, right)
        @test size(res) == (1,2)
        @test isapprox(res[1,1], 0.45)
        @test isapprox(res[1,2], 0.09258200997725517)
    end

    @testset "cantelli" begin
        Random.seed!(42)
        for _ in 1:100
            X = 100
            Y = 100
            i1 = rand(X)
            i2 = .5 .+ 2 .* rand(Y)
            c11 = cantelli(i1, Statistics.mean(i1), Statistics.std(i1))
            c12 = cantelli(i1, Statistics.mean(i2), Statistics.std(i2))
            # println(c11 - c12)
            @test all(c11 .>= 0)
            @test all(c12 .>= 0)
            @test sum(c12) < sum(c11)
        end
    end

    @testset "unionmask" begin
        A = zeros(50, 50)
        B = zeros(50, 50)
        A[2:4, 2:4] .= 1
        B[3:5, 3:5] .= 1
        ABU = unionmask(A, B)
        @test all(ABU[A .== 1] .== 1)
        @test all(ABU[B .== 1] .== 1)
    end

    @testset "distancenmask" begin
        A = zeros(50, 50)
        B = zeros(50, 50)
        A[2:4, 2:4] .= 1
        A[6:6, 6:6] .= 1
        B[3:5, 3:5] .= 1
        Acc = Images.label_components(A,trues(3,3))
        Bm = ERGO.tomask(Images.label_components(B,trues(3,3)))
        ABU = compute_distances_cc_to_mask(Acc, Bm)
        @test minimum(ABU) == 0
        @test maximum(ABU) == sqrt(2)
    end

    @testset "mahabrefactor" begin
        Random.seed!(42)
        for _ in 100
            N = 100
            k = 10
            data = Float64.(10  .+ randn(N, k) .* 100)
            zs2, ms2, ico = computemahabdistances(data, Statistics.mean(data, dims=1), Statistics.cov(data))
            zs, ms, ico = computemahabdistances_vector(data, data, Statistics.mean(data, dims=1), Statistics.cov(data))
            @test ms2 == ms
            @test zs2 == zs
        end
    end

    @testset "meancov" begin
        Q = zeros(Float64, 3, 3, 3)
        Q[1, :, :] = ones(eltype(Q), 3,3)
        Q[2, 2, 2] = 42.0
        rs = meancov(Q)
        @test isapprox(rs[1,2,2] , (42.0+1.0)/3.0)
    end

    @testset "mahalanobiscomposite" begin
        a = zeros(2,2)
        a[1,1] = 1.0
        a[2,2] = 1.0
        Random.seed!(42)
        for _ in 1:100
            b = rand(10, 2)
            b2 = rand(10, 2)
            euc = sqrt(sum((b2[1,:].-b[1,:]).^2))
            m = Statistics.mean(b, dims=1)
            Zs, ms = computemahabdistances(b, m, Statistics.cov(b))
            @test isapprox(ms, m)
            Zps, ms, co = computemahabdistances_vector(b, b, m, Statistics.cov(b))
            @test isapprox(Zs, Zps)
            Zps, ms, co = computemahabdistances_vector(b, b2, Statistics.mean(b2, dims=1), Statistics.cov(b2))
            @test ! isapprox(Zs, Zps)
        end
    end
end
