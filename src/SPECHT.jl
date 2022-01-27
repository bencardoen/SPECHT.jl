
# Copyright (C) 2018-2022 Ben Cardoen bcardoen@sfu.ca
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

module SPECHT

import DataFrames
import CSV
import ERGO
using Logging
import Images
import Random
import Glob
import Statistics
import ImageFiltering
import Distributions
using LinearAlgebra
import ImageMorphology
import StatsBase
using Match


export computemahabdistances, computemahabdistances_vector, contrast_x_to_y, rlap, scale, find_th,
checkdir, glg, process_tiffimage, processimg, glg, compute_nl, process_tiff,
compute_distances_cc_to_mask, quantify_spots_mitophagy_new, qspot, harmonicmean,
unionmask, meancov, normalize_stat_dist, cantelli, quantify_spots_mitophagy_new,
annotate_spots, mahalanobis, filterth, intersectimg, filter_k, dice_jaccard,
jaccard, unionimg, filter_cc_sqr_greater_than, cnl, computeintensityboundary,
quantify_adjacent_mitophagy, process_cell, computemasks, lowsnrsegment, validatechannels,
cohen_d, quantify_adjacent_c12, describeimg, tcolors, quantify_nearest_mitophagy,
heavyedge, nz, ❗0, dropartifacts, rlap!, segmentnondecon, iterative, id, 𝛁, ⨣, keep_positive,
process_non_segmented_non_decon_channels, visout, record_stats, ♌, maskoutline,
⨥, csum, 🌄, tricoloroutline, cycle_vec_1, checkdrift, checksegmentsaligned, filtercomponents_using_maxima_IQR,
quantify_c12_mito_overlap, aniso2, cellstats, score_masks_mcc, score_masks_visual, poissonnoise,
sandp, gaussiannoise, vpdf, fastgaussian2d, computelogotsu, computeotsu, generate_scenario, coordstogt,
generate_rand_coordinates



"""
	generate_rand_coordinates(X, Y, N; offset=0, seed=nothing)
	Generate N 2D Float64 coordinates in [1:X, 1:Y].
	Offset > 0 forces coordinates to be placed offset away from border.
	!isnothing(seed) will seed RNG
	Returns Nx2 array.
"""
function generate_rand_coordinates(X, Y, N; offset=0, seed=nothing)
	if ! isnothing(seed)
		Random.seed!(seed)
	end
	rs = rand(N, 2)
	if offset != 0
		@assert 0 < offset <= min(X, Y)
		rs[:,1] .*= (X-offset*2)
		rs[:,2] .*= (Y-offset*2)
		rs .+= offset
	else
		rs[:,1] .*= X
		rs[:,2] .*= Y
	end
	rs
end

"""
    contrastive(Xs, Ys)
    Given 2 arrays of Mxk, Nxk, where k is the nr of features for each object (row).
	Lets Xs have a set level label of 'X', and Ys a set level label of 'Y'.
	Compute the plausibility that for each x ∈ X, x → Y
"""
function contrast_x_to_y(xs, ys)
	@assert(size(xs, 2) == size(ys, 2))
	μy = Statistics.mean(ys, dims=1)
	ρy = Statistics.cov(ys)
	zx = computemahabdistances_vector(xs, ys, μy, ρy)[1]
	zy = computemahabdistances_vector(ys, ys, μy, ρy)[1]
	ρx = cantelli(zx, Statistics.mean(zy), Statistics.std(zy))
	return ρx
end


"""
	coordstogt(coords, X, Y)
	Create a 2D binary mask I where I[round(c_x), round(c_y)]=1 for each c in coords.
"""
function coordstogt(coords, X, Y)
	a = zeros(X, Y)
	for c in coords
		x, y = Int.(round.(c))
		a[x, y] = 1
	end
	ERGO.tomask(a)
end

"""
	generate_scenario(X, Y, nbright, ndim; seed=42, σ=3, offset=32, dimfactor=4)
	Create an image of X x Y pixels, with bright and dims spots.
	Spots are isotropic 2D Gaussians of σ and σ*2 respectively.
	Dimfactor controls the brightness factor of bright v dim.
	Returns the binary mask of GT locations, image, coordinates for bright and dim
"""
function generate_scenario(X, Y, nbright, ndim; seed=42, σ=3, offset=32, dimfactor=4)
	if seed != 0
		Random.seed!(seed)
	end
	N = ndim
	cv = [[σ 0; 0 σ] for _ in 1:N]
	rs = generate_rand_coordinates(X, Y, N; offset=offset)
	GT = coordstogt([rs[i,:] for i in 1:N], X, Y)
	G = fastgaussian2d([rs[i,:] for i in 1:N], cv, X, Y)
	N2= nbright
	cv = [[2*σ 0; 0 2*σ] for _ in 1:N2]
	rs2 = generate_rand_coordinates(X, Y, N2; offset=offset*4)
	GT2 = coordstogt([rs2[i,:] for i in 1:N2], X, Y)
	G2 = fastgaussian2d([rs2[i,:] for i in 1:N2], cv, X, Y)
	ground = ERGO.normimg(GT .+ GT2)
	total = ERGO.normimg(G./dimfactor .+ G2)
	return ground, total, rs, rs2
end




"""
	score_masks_visual(gt_mask, pr_mask)
	Return an image where
		FP = red
		TP = green
		FN = blue
		for each component in pr_mask.
"""
function score_masks_visual(gt_mask, pr_mask)
	fp, tp, fn = ERGO.aszero(gt_mask), ERGO.aszero(gt_mask), ERGO.aszero(gt_mask)
	gt_ccs = Images.label_components(gt_mask)
	pr_ccs = Images.label_components(pr_mask)
	gt_inds = Images.component_indices(gt_ccs)[2:end]
	pr_inds = Images.component_indices(pr_ccs)[2:end]
	for gt_i in gt_inds
		if any(pr_mask[gt_i] .> 0)
			tp[gt_i] .= 1
		else
			# Not covered
			fn[gt_i] .= 1
		end
	end
	for pr_i in pr_inds
		if sum(gt_mask[pr_i]) == 0
			fp[pr_i] .= 1
		end
	end
	return fp, tp, fn
end

"""
	computelogotsu(raw, sigmas=[σ, 2*σ])
	Compute LoG blob detection (scale space) from σ to 2*σ, followed by Otsu tresholding to remove Fps.
	A 4px border is added to located centroids.
"""
function computelogotsu(raw, σ=3)
	blg = Images.blob_LoG(raw,[σ,2*σ])
	mk = Float64.(ERGO.aszero(raw))
	for b in blg
		x = b.location[1]
		y = b.location[2]
		mk[x,y] = b.amplitude
	end
	mkmsk = copy(mk)
	mkmsk[mk .> 0] .= 1
	thx = Images.otsu_threshold(mk)
	mkx = copy(mk)
	mkx[mk .< thx] .= 0
	mkx[mk .>= thx] .= 1
	return mkx, maskoutline(iterative(mkx, ImageMorphology.dilate, 4))
end

"""
	computeotsu(raw)
	Simple otsu threshold binary mask + dilation (2px)
"""
function computeotsu(raw)
	ot = Images.otsu_threshold(raw)
	iq = copy(raw)
	iq[iq.<ot] .= 0
	iq[iq.>0] .= 1
	return iq, maskoutline(iterative(iq, ImageMorphology.dilate, 2))
end

"""
	vpdf(pdfs, coords)

Return sum of pdf on coord
"""
function vpdf(vs, coords)
	return sum(Distributions.pdf(v, coords) for v in vs)
end

"""
	fastgaussian2d(centers, covs, XN, YN)
Return an XN by YN array where each [x, y] = sum(pdf_gaus(center, cov)[x,y])
|centers| == |covs|, for each pair a Gaussian pdf is generated
"""
function fastgaussian2d(centers, covs, XN, YN)
	vs =  [Distributions.MvNormal(center, cv) for (center, cv) in zip(centers, covs)]
	Q = (vpdf(vs, [x, y]) for x in 1:XN,  y in 1:YN)
	return Q |> collect
end
"""
	electronic noise modelled by Gaussian μ 0, σ.
	Set bits to encoding of image.
"""
function gaussiannoise(img, σ; μ=0, bits=8)
	return abs.(rand(Distributions.Normal(0, σ), size(img))./(2^bits))
end

"""
	chechsegmentsaligned
	Given two segments (binary masks of same dimensions), check if they share
	a non zero overlap. If not the multichannel segmentation has failed.
"""
function checksegmentsaligned(segments)
	return any(reduce(.*, segments) .> 0)
end

"""
	aniso2(Q)
	For a 2D matrix Q
	Returns A, λ1, λ2, where A = 0 if isotropic, else --> 1
"""
function aniso2(Q)
	λ1, λ2 = imgpca2(Q)
	if λ1 == λ2 == 0
		# Point ~ iso
		return 0, λ1, λ2
	end
	return 1 - λ2/λ1, λ1, λ2
end

function cellstats(cellgraymask)
	CM = ERGO.tomask(cellgraymask)
	area = length(CM[CM.>0])
	aniso, λ1, λ2 = aniso2(cellgraymask)
	return Dict(:area=>area, :anisotropy=>aniso, :λ1=>λ1, :λ2=>λ2)
end



function imgpca2(img)
	@assert(! all(img .== 0))
	if length(img[img .> 0]) == 1
		return [0. , 0.] #Point
	else
		rs = imgmoment2D(img)
		ctr = Statistics.mean(rs, dims=1)
		Rc = rs .- ctr
		U, S, V = Statistics.svd(Rc)
		if size(Rc)[1] <= 2
			return S[1], 0.0
		end
		return S
	end
end

function imgmoment2D(img)
	X, Y = size(img)
	zr = zero(eltype(img))
	N = length(img[img .> zr])
	@debug "Have $N non zero voxels"
	res = zeros(Float64, N, 2)
	c = 1
	for x in 1:X, y in 1:Y
		i = img[x,y]
		if i > zr
			res[c,:] = [x*i, y*i]
			c += 1
		end
	end
	return res
end

"""
	Detect if a (cell) mask is moving out of frame
"""
function checkdrift(mask, edgetolerance=1)
	@assert edgetolerance>0
	X, Y = size(mask)
	top = mask[1:edgetolerance, :]
	bottom = mask[end-edgetolerance+1:end, :]
	left = mask[:, 1:edgetolerance]
	right = mask[:, end-edgetolerance+1:end]
	sums = [sum(t) for t in [top, bottom, left, right]]
	# cm = copy(mask)
	# cm[1+edgetolerance:X-edgetolerance, 1+edgetolerance:Y-edgetolerance] .= mask[1+edgetolerance:X-edgetolerance, 1+edgetolerance:Y-edgetolerance] .* 0
	# return any(cm .> 0)
	return any(sums .> 0)
end

"""
	Use IQR of local maxima as a heuristic to reject false positives.
	Sigma controls the pre-smoothing (for low SNR/pixellated images)
	Do not set > ~ 1 or so
	Mask is a binary object mask of candidate objects
	Raw image, well, naming...
"""
function filtercomponents_using_maxima_IQR(raw_nsegmented_image, mask; sigma=0.25)
	ccs = Images.label_components(mask)
	raw_segmented_image = ImageFiltering.imfilter(raw_nsegmented_image, ImageFiltering.Kernel.gaussian((0.25, 0.25)))
	MXF = ERGO.aszero(raw_segmented_image)
	Q3 = StatsBase.quantile(❗0(raw_segmented_image), 0.75)
	Q1 = StatsBase.quantile(❗0(raw_segmented_image), 0.25)
	IQR = Q3-Q1
	TH = Q3 + 1.5*IQR
	@assert Q1 > 0
	# @assert maximum(❗0(raw_segmented_image)) > TH
	@debug "Q1 $Q1 Q3 $Q3 IQR $IQR TH $TH"
	for ix in Images.findlocalmaxima(raw_segmented_image)
		maxval = raw_segmented_image[ix]
		if any(maxval .> TH)
			MXF[ix] = maxval
		end
	end
	mk = dropccs(ccs, MXF)
	return mk
end

"""
	dropccs(comps, msk_to_keep)
	Return a mask where components present in msk_to_keep are retained, remainder dropped.
"""
function dropccs(ccs, msk)
	res = ERGO.aszero(msk)
	indices = Images.component_indices(ccs)[2:end]
	for ind in indices
		if any(msk[ind] .> 0)
			res[ind] .= 1
		end
	end
	return res
end


"""
	tricoloroutline(rawimages, maskedimages)
	Returns a 3 color image over raw image overlayed with outline of mask
"""
function tricoloroutline(raws, masks)
	return tcolors([🌄(r) .⨥ ♌(o) for (r,o) in zip(raws, masks)])
end

"""
	visout(yellow, mitomask, mitoimg)
	Return overlay of c1c2 mask on mito raw, outline of yellow / mito on mito raw
"""
function visout(unionc1c2image, mitomask, mitogrey)
	ym_on_mito = tcolors([mitogrey .⨥ unionc1c2image, mitogrey .⨥ unionc1c2image, mitogrey])
	yo = ♌(unionc1c2image)
	bo = ♌(mitomask)
	m_b = tcolors([mitogrey.⨥yo, mitogrey.⨥yo, mitogrey .⨥ bo])
    return ym_on_mito, m_b
end

function cycle_vec_1(vec)
	if length(vec) > 1
		c = copy(vec[2:end])
		push!(c, vec[1])
		return c
	else
		return vec
	end
end


function csum(a, b)
	t = eltype(a)
	s = Float64(a) + Float64(b)
	s = min(s, 1)
	return t(s)
end

⨥ = (x, y) -> csum(x, y)

function keep_positive(e)
	c = copy(e)
	c[e .< zero(eltype(e))].=zero(eltype(e))
	return c
end

⨣ = keep_positive

𝛁 =  ImageMorphology.morpholaplace

id = x -> identity(copy(x))

"""
	segmentnondecon(image, iterations=3, quantile=0.75)
	Segments image into 1 cell, under the prior that only 1 complete cell is in field of view (other partial cells can exist in fov).
	Iterations should be set at ceil(precision in pixels).
	Say precision is 160nm, pixelsize=38nm, then set iterations to 4.
	This ensures holes in the mask <4 pixels are closed, and adds a sufficient border to prevent loss of information.
	Image is assumed to be non-deconvolved, with a pretty low SNR.
	Quantile should be set in function of field of view and size (expected/average) of cell.
	For example, fov of 100x100pixels, with cell size 30x30 with low SNR, then quantile ~ 0.75 is a safe choice, iow quantile should be function of (1 - ratio cell size / fov)
"""
function segmentnondecon(nimg, iters=3, quantile=0.75)
	cimg = copy(nimg)
	Q = StatsBase.quantile(cimg[:], quantile)
	ft = copy(cimg)
	ft[cimg .< Q] .= 0
	## 3 stage sweep
	ft1 = SPECHT.denoisemd(ft, 1)
	ftm2 = SPECHT.denoisemd(ft1, 2)
	ftm3 = SPECHT.denoisemd(ftm2, 3)
	## Find largest component ~ cell
	ccs = Images.label_components(ERGO.tomask(ftm3))
	if maximum(ccs) < 1
		@debug "No object"
		return nothing
	end
	lns = Images.component_lengths(ccs)[2:end]
	sz, i = findmax(lns)
	Q = ERGO.aszero(cimg)
	Q[Images.component_indices(ccs)[2:end][i]] .= 1
	## Dilate to make sure we get sufficient border
	Q = ERGO.tomask(iterative(Q, ImageMorphology.dilate, iters))
	# If holes are present < precision, close them
	Q = iterative(Q, ImageMorphology.closing, iters*2)
	if checkdrift(Q, iters)
		@debug "Cell drifting out of view"
		return nothing
	end
	return Q
end

"""
	Dedicated function for cases where cell is not segmented, with partial cells in view.
	input should be 3 (channels) x 2D (image) array. PRC controls precision/recall detection, quantile segmentation (higher == stricter)
"""
function process_non_segmented_non_decon_channels(slices; PRC=1, sigma=1, quantile=0.9, pixprec=3)
	@debug "PRC = $PRC σ = $sigma QX = $quantile PX = $pixprec"
	# Segment per channel
	@assert(length(slices) == 3)
	@assert(PRC > 0)
	segments = [segmentnondecon(slice, pixprec, quantile) for slice in slices]
	if any(isnothing.(segments))
		@debug "Segmentation failed!"
		return nothing
	end
	if ! checksegmentsaligned(segments)
		@debug "Segments not aligned!"
		return nothing
	end
	# Fuse to ensure we get everything (especially for mito)
	all3 = ERGO.tomask(reduce(.+, segments))
	alledge = ⨣(𝛁(all3))
	edges = [⨣(𝛁(s)) for s in segments]
	results = Dict()
	for (i, cimg) in enumerate(slices)
		channelnr = (i-1)
		SEGMENTED = cimg.*all3
		em = iterative(edges[i], ImageMorphology.dilate, pixprec*2)
		_em = em
		# if channelnr == 0
		_em = nothing
		# end
		# Z to 42 to make it clear we're autotuning
		ccs, imgl, Tg, _img, msk = process_tiffimage(SEGMENTED, 42, [sigma, sigma], true, PRC, 0, edgemask=_em)
		### Mito labelling has an edge at the edge of mito, so non issue.
		### The other channels are different story.
		### Mask edges from the computation to not skew it
		# if channelnr != 0
		# 	msk[em .== 1] .= 0
		# end
		ccs = Images.label_components(msk)
		SF = dropartifacts(ccs, SEGMENTED)
		# if channelnr != 0
		# 	SF[em .== 1] .= 0
		# end
		ccs = Images.label_components(ImageMorphology.closing(SF))
		ccs = Images.label_components(SF)
		FILTERED = filter_cc_sqr_greater_than(ccs, SEGMENTED, pixprec)
		## Add pixel to enable outline
		FT = ImageMorphology.dilate(iterative(FILTERED, ImageMorphology.closing, pixprec))
		outline = ⨣(𝛁(FT))
		results[channelnr] = SEGMENTED, FT, outline, em, cimg, segments[channelnr+1]
	end
	return results
end

🌄 = ERGO.normimg

"""
	For a binary mask, return the edge (outline).
"""
function maskoutline(msk)
	return ⨣(𝛁(msk))
end

♌ = maskoutline

"""
	poissonnoise(img, λ)
"""
function poissonnoise(img, λ; bits=8)
	@assert bits > 0
	mx = 2^bits
	@assert 0 < λ < mx
	bg = rand(Distributions.Poisson(λ), size(img)) ./ mx
	return bg
end

"""
	sandp(grade, img)
	Apply salt and pepper noise to img, with 1-grade probability of a pixel being changed.
	Models sensor error
"""
function sandp(grade, img)
	X, Y = size(img)
	ps = rand(X,Y)
	rn = rand(X,Y)
	rn[rn .> 0.5] .= 1
	rn[rn .<= 0.5] .= 0
	rn = ERGO.tomask(rn)
	IQ = copy(img)
	IQ[ps .> grade] .= img[ps .> grade].*rn[ps .> grade]
	return IQ, ps
end

"""
	zeta(N, z)

	Compute up to N terms the z'th zeta function.
"""
function zeta(N, z)
    sum(1/i^z for i in 1:N)
end

"""
	aperyapprox(N)

	Approximate the Apery constant with up to N terms.
	Note: naive implementation, faster ones exist.
	Used in versioning SPECHT
"""
function aperyapprox(N)
	return zeta(N, 3)
end


"""
	quantify_c12_mito_overlap(C12mask, mitomask, raw_mito_img; pxmargin)
	For n ∈ N C12 components, return
		- M, μ, σ
		Where M is nr of > 0 pixels in mitomask under C12spot, μ, σ are intensities under that spot
	Px marging dilates each spot with k pixels to compensate for precision
"""
function quantify_c12_mito_overlap(c1c2_img, mito_img, raw_mito_img; pxmargin=0)
	mito_mask = ERGO.tomask(mito_img)
	c1c2_coms = Images.label_components(ERGO.tomask(c1c2_img), trues(3,3))
	C1C2_N = maximum(c1c2_coms)
	@debug "Got $(C1C2_N) components"
	result = zeros(Float64, C1C2_N, 3)
	for (nth, indexn) in enumerate(Images.component_indices(c1c2_coms)[2:end])
		C12 = ERGO.aszero(c1c2_img)
		C12[indexn] .= 1
		if pxmargin > 0
			C12 = SPECHT.iterative(C12, ImageMorphology.dilate, pxmargin)
		end
		npx = sum(mito_mask .* C12)
		if npx == 0
			μ, σ = 0, 0
		else
			mito_under_spot = (mito_mask .*raw_mito_img)[C12 .> 0]
			μ, σ = Statistics.mean(Float64.(mito_under_spot)), Statistics.std(Float64.(mito_under_spot))
		end
		result[nth, :] .= npx, μ, σ
	end
	return result
end

"""
	iterative(input, function=id, iterations=1)
	Apply function iteration times to input
"""
function iterative(m, f=id, iters=1)
	if iters == 0
		return m
	end
	return iterative(f(copy(m)), f, iters-1)
end

function keepyellow(tif)
    image = copy(tif)
    CO = ERGO.aszero(image)
    # yellow =  Images.RGB{Images.N0f16}.( 1, 1, 0)
    CO[(Images.red.(image) .== 1) .& (Images.green.(image) .== 1)] .= Images.RGB{Images.N0f16}.( 1, 1, 0)
    return CO
end

function quantify_nearest_mitophagy(c1c2_mask, mito_mask, mito_img)
	# Find the min distance to nearest mito from C1C2, its area, and its mean intensity
	c1c2_coms = Images.label_components(c1c2_mask, trues(3,3))
	mito_coms = Images.label_components(ERGO.tomask(mito_mask), trues(3,3))
	N = maximum(c1c2_coms)
	M = maximum(mito_coms)
	@debug "Have $N C1C2 and $M Mito"
	if N == 0
		@debug "No C1C2 components"
		return nothing, ERGO.aszero(mito_img)
	end
	if M == 0
		@debug "No Mito components"
		res = zeros(Float64, N, 3)
		res[:, 1] .= Inf
		return res, ERGO.aszero(mito_img)
	end
	c1c2indices = Images.component_indices(c1c2_coms)[2:end]
	mito_indices = Images.component_indices(mito_coms)[2:end]
	mito_lengths = Images.component_lengths(mito_coms)[2:end]
	if length(mito_lengths) == 1
		@warn "1 mito only, suspect, please check mask"
	end
	results = zeros(Float64, N, 3)
	cm = ERGO.aszero(c1c2_mask)
	c1c2res = ERGO.aszero(c1c2_mask)
	mitores = ERGO.aszero(c1c2_mask)
	for (ic1c2, c1c2indices) in enumerate(c1c2indices)
		cm[c1c2indices] .= 1
		distance_map = Images.distance_transform(Images.feature_transform(Bool.(cm)))
		mindist, minindex = findmin([minimum(distance_map[mito_ind]) for mito_ind in mito_indices])
		mito_int = mito_img[mito_indices[minindex]]
		μ = Statistics.mean(mito_int)
		area = mito_lengths[minindex]
		results[ic1c2, :] .= [mindist, μ, area]
		cm[c1c2indices] .= 0
		c1c2res[c1c2indices] .= ic1c2/N
		mitores[mito_indices[minindex]] .= ic1c2/N
	end
	tricolor = tcolors([c1c2res, c1c2res, mitores])
	return results, tricolor
end

"""
	μ, med, σ, min, max of 1XN array
"""
function describeimg(_xs)
	xs = Float64.(_xs) # Force promotion, Image types > 1.7 are warning for deprecation, and F64 is more accurate.
	xs = xs[xs .> 0]
	if length(xs) == 0
		@debug "No data to compute statistics for"
		return [0.0 for _ in 1:5]
	end
	return Statistics.mean(xs), Statistics.median(xs), Statistics.std(xs), minimum(xs), maximum(xs)
end

"""
	Slice an array to retain only positive elements.
"""
function nz(Xs)
	return Xs[Xs.>0]
end

❗0 = nz

"""
	Given output of Specht's object detection (ccs), remove low intensity objects using stat test
"""
function dropartifacts(ccs, segmented)
	allindices = Images.component_indices(ccs)[2:end]
	base = ❗0(segmented)
	cohens = [indices for indices in allindices if cohen_d(segmented[indices], base) > 0]
	SF = ERGO.aszero(segmented)
	for ind in cohens
		SF[ind] .= 1
	end
	return SF
end

"""
	cohend_d(xs, ys)
	Return for N, M x 1 arrays the Cohen D (effect size)
	Returns NaN if N+M-2 <= 0
	! Not symmetric, cohen_d(xs, ys) == -cohen_d(ys,xs)
"""
function cohen_d(xs, ys)
	μx, μy = Statistics.mean(xs), Statistics.mean(ys)
	nx, ny = length(xs), length(ys)
	denom = nx+ny-2
	if denom == 0
		return NaN
	end
	σpool = √((sum((xs .- μx).^2) + sum((ys .- μy).^2))/denom)
	return (μx - μy)/σpool
end

"""
	Return left ∪ right
"""
function intersectimg(left, right)
	@assert(size(left) == size(right))
	res = ERGO.aszero(left)
	Z = zero(eltype(left))
	One = one(eltype(left))
	res[(left .> Z) .& (right .> Z)] .= One
	return res
end

"""
	Return a mask where connected components of img are > SQ in both X and Y.
"""
function filter_cc_sqr_greater_than(ccs, img, SQ)
    msk = ERGO.aszero(img)
    indices = Images.component_indices(ccs)[2:end]
    boxes = Images.component_boxes(ccs)[2:end]
    for (i, _ind) in enumerate(indices)
        b = boxes[i]
        XYRAN = (b[2] .- b[1]) .+ 1
        if XYRAN[1] > SQ &&  XYRAN[2] > SQ
            msk[_ind] .= 1
        end
    end
    return msk
end

"""
	Return the mean covariance matrix of an array of covariance matrices
"""
function meancov(covs)
    _n, _r, _c = size(covs)
    @assert _n > 1
    @assert _r == _c
    @assert _r > 1
    return sum(covs, dims=1) ./ Float64(size(covs, 1))
end

"""
	For a 3 channel cell, where results are stored in dict res, compute statistics based on 3x 2D masks.
"""
function quantify_spots_mitophagy(res)
	@warn "DEPRECATED"
    _, mask_mito = res[0]
    cc1, mask_1 = res[1]
    cc2, mask_2 = res[2]
    c1count = maximum(cc1)
    c2count = maximum(cc2)
    c1_c2 = copy(mask_1 .* mask_2)
    c1_c2_coms = Images.label_components(c1_c2, trues(3,3))
    c1c2count = maximum(c1_c2_coms)
    overlap_mito = copy(c1_c2 .* mask_mito)
    overlap_mito_coms = Images.label_components(overlap_mito, trues(3,3))
    count_overlap_mito = maximum(overlap_mito_coms)

    overlap_c1_mito = mask_1 .* mask_mito
    overlap_c2_mito = mask_2 .* mask_mito
    overlap_c1_mito_coms = Images.label_components(overlap_c1_mito, trues(3,3))
    overlap_c2_mito_coms = Images.label_components(overlap_c2_mito, trues(3,3))
    count_overlap_c1_mito = maximum(overlap_c1_mito_coms)
    count_overlap_c2_mito = maximum(overlap_c2_mito_coms)
    _ccs = [ cc1, cc2, c1_c2_coms, overlap_mito_coms, overlap_c1_mito_coms, overlap_c2_mito_coms ]
    areas = [Images.component_lengths(_cc)[2:end] for _cc in _ccs]

    return c1count, c2count, c1c2count, count_overlap_mito, count_overlap_c1_mito, count_overlap_c2_mito, areas
end

"""
    Compute Dice, Jaccard indices for arguments.
"""
function dice_jaccard(img1, img2)
	@assert(size(img1) == size(img2))
	j = jaccard(img1, img2)
	d = (2*j) / (1+ j)
	return d, j
end

"""
    Return |A ∩ B|/| A ∪ B |
"""
function jaccard(a, b)
	@assert(size(a) == size(b))
	z = zero(eltype(a))
	m1 = a .> z
	m2 = b .> z
	u = unionimg(m1, m2)
	i = intersectimg(m1, m2)
	return count(i) / count(u)
	# return count((m2) .& (m1)) / count((m1) .| (m2))
end

"""
	Return union (pixelbased) of two images, left argument determines type.
"""
function unionimg(left, right)
	res = ERGO.aszero(left)
	Z = zero(eltype(left))
	One = one(eltype(left))
	res[(left .> Z) .| (right .> Z)] .= One
	return res
end


"""
	For a given image with connected components and a matching array csvcol, return a mask where img[component_i] = csvcol[i]
"""
function annotate_spots(img, concoms, csvcol)
    NL = maximum(concoms)
    indices = Images.component_indices(concoms)[2:end]
    mask = copy(Images.N0f16.(img))
    mask[concoms .== 0] .= zero(eltype(mask))
    # counts = countmap(concoms[concoms .> 0])
    for component in 1:NL
        @inbounds ind = indices[component]
        # class_nr = csvcol[component]
        @inbounds mask[ind] .= csvcol[component]
    end
    return mask
end



"""
    Normalize a distance vector to a mean and std target
    Return abs(dist - μ)/σ
"""
function normalize_stat_dist(dists, μ, σ)
    return (abs.(dists .- μ))./(σ)
end


"""
    Transform a vector to a probability using Cantelli's theorem. Normalize using μy, σy of a given target distribution.
    Outputs a score close to 1 for very similar values, 0 for dissimilar.
    Pr[Z_i ≧ z] = 1 / (1+z^2)
"""
function cantelli(xs, μy, σy)
    nx = normalize_stat_dist(xs, μy, σy)
    px = 1 ./ (1 .+ nx.^2)
    return px
end

"""
	rlap(array)
    Return the absolute value of the negative part of the array.
	Used in LoG detection.
"""
function rlap(lap)
    return rlap!(copy(lap))
	z = zero(eltype(lap))
    lp[lp .> z] .= z
    lp .= z .- lp
    return lp
end

"""
	rlap!(array)
    Return the absolute value of the negative part of the array.
	Used in LoG detection.
"""
function rlap!(lap)
	z = zero(eltype(lap))
	_f = x ->  x < z ? -x : z
	return map!(_f, lap, lap)
end

"""
	Utility function, if dir does not exists, make the path (including parents) s.t. dir exists.
"""
function checkdir(dir)
    if ! isdir(dir)
        mkpath(dir)
    end
end

"""
	Compute scaling factor based on distribution of values in array. If array ~ N(μ, σ) likely returns 0.
"""
function scale(array)
	k = Distributions.kurtosis(Float64.(array))
	if k < 0.0
		@warn "Negative adjusted kurtosis, clamping from $(k) to 0"
		return 0.0
	end
	return k^(.25)
end

"""
    For a 2D array img, return an array of indices where img[x,y] < th.
	Given that an image .> 0 in most encodings, having th < 0 is not meaningful.
"""
function filterth(img, th)
    sizex, sizey = size(img)
    ft = Vector{CartesianIndex{2}}()
	## Todo rewrite with functional map + product
    for x in 1:sizex
        for y in 1:sizey
            if img[x,y] < th
                push!(ft, CartesianIndex{2}(x,y))
            end
        end
    end
    return ft
end


"""
	Apply to image im a median filter with window w*2+1
"""
function denoisemd(im, w)
	@debug "Median filter"
    mg= Images.mapwindow(Statistics.median, im, [w*2+1 for _ in 1:length(size(im))])
    mg[isnan.(mg)] .= zero(eltype(im))
	return mg
end

"""
	lowsnrsegment(img, edgeiters=5)
	Given a low SNR image, single channel, with 1 cell inside, apply a basic segmentation to recover the cell
	Returns masked img, mask, and edge mask
	Edgeiters control how dilated the edge mask is
	sigmas : blurring
"""
function lowsnrsegment(img; edgeiters=20, sigmas=(2,2))
	median = denoisemd(Images.N0f8.(img), 1)
	median2 = denoisemd(median, 2)
	### Blur to be a bit less conservative
	smoothedmedian = ImageFiltering.imfilter(median2, ImageFiltering.Kernel.gaussian((sigmas)))
	### Select the cell as the largest component
	MASK = ERGO.tomask(smoothedmedian)
	CCS = Images.label_components(MASK)
	lengths = Images.component_lengths(CCS)[2:end]
	maxcomp = findall(lengths .== maximum(lengths))
	@assert(length(maxcomp) == 1)
	cellmask = ERGO.aszero(MASK)
	cellimg = ERGO.aszero(MASK)
	indices = Images.component_indices(CCS)[2:end][maxcomp][1]
	cellimg[indices] .= img[indices]
	cellmask[indices] .= oneunit(eltype(img))
	edgemask = heavyedge(cellmask, edgeiters)
	return cellimg, cellmask, edgemask
end

"""
	heavyedge(mask, iters)
	Given a binary mask, return the dilated (iters times) edge.
	Used in masking a cell boundary.
"""
function heavyedge(mask, iters=5)
	img_morpholap = ImageMorphology.morpholaplace(mask)
	d = ImageMorphology.dilate(img_morpholap)
	for i in 1:(iters-1)
		d = ImageMorphology.dilate(d)
	end
	d[d .< 0] .= 0
	return ERGO.tomask(d)
end

"""
	Find an autoscaling threshold for the negative laplacian (imgl).
	If auto = true, determine a threshold automatically using the kurtosis.
	You can tune the threshold in kurtosis space by setting scale < 1 (precision), or > 1 (recall).
	Z is used if auto=false, else k^(1/4)/scale
	Treshold = μg * σg^Z (geometric z-scaling)
	Return the threshold and negative part of the first argument.
	edgemask = For cells where the outline of the cell leads to high intensity change (gradient), add a mask that delineates that boundary, then use that to exclude it from the tuning stage
"""
function find_th(imgl, Z, auto=false, scale=1, edgemask=nothing)
	negl = rlap(imgl)
	if ! isnothing(edgemask)
		@debug "Using edgemask"
		@assert size(edgemask) == size(imgl)
		negl[edgemask .> 0 ] .= 0
	end
    Zp = Z
	avals = Float64.(negl[:])
    if auto == true
		kurt = Distributions.kurtosis(avals)
        Zp = kurt^(.25)/scale
        @debug "Using self scaling Z of $(Zp)"
    end
    gem, ges = ERGO.gmsm(avals)
    Tg = gem * ges^Zp
    return Tg, negl
end


"""
    computemahabdistances_vector
    Compute the Mahalanobis distance, for each row of an NxK matrix, to the distribution inferred from data.
    If means and covariance are not provided, compute them. If data is composition of distributions (say objects from cells), then a bootstrapped mean/cov is more accurate.
"""
function computemahabdistances_vector(from, to, ms, cv)
    # K samples of N dimenions
    data = from
    @assert(length(size(data))==2)
    K, N = size(data)
    @assert(size(to, 2) == size(from, 2))
    means = ms
    @assert(size(means) == (1,N))
    Zs = zeros(eltype(data), K)
    co = cv
    @assert(size(co) == (N, N))
    ico = LinearAlgebra.inv(co)

    for k in 1:K
        row = reshape(data[k, :], 1, :)
        Zs[k] = mahalanobis(row, means, ico)
    end
    return Zs, means, co
end

"""
	Compute the mahalanobis distance for vectors u, v, and inverse covariance.
"""
function mahalanobis(u, v, icov)
    # 1xN * NxN * N-1
    return sqrt((u-v) * icov * (u-v)')[1]
end

"""
	Compute mahalanobis distances for each vector in array data to itself, with given means and covariance.
"""
function computemahabdistances(data, ms, cv)
    return computemahabdistances_vector(data, data, ms, cv)
end

"""
    process_tiff(file, z, sigmas, selfscale, precisionrecall, smooth)

Find spots in a tiff `file` (2D). `Z `determines the z-test threshold, `sigmas` is an iterable of 2 Gaussian sigmas for pre/postprocessing.
`selfscale` (Boolean) ignores `z` and uses heuristics that will align the results over multiple tiffs consistently. If `selfscale`, `precisionrecall` determines on which side to err.
A value > 1 increases recall. If `smooth` != 0, a final smoothing of the masks is produced.

Returns connected components, 2nd diff, computed threshold, image and mask.

"""
function process_tiff(file, z, sigmas, selfscale, tim, smooth, selfscalemethod="kurtosis", edgemask=nothing)
	img = Images.load(file);
	return process_tiffimage(img, z, sigmas, selfscale, tim, smooth)
end

"""
	Same as process_tiff, but from an already loaded image.
"""
function process_tiffimage(img, z, sigmas, selfscale, PRC, smooth; selfscalemethod="kurtosis", edgemask=nothing)
	if selfscale && selfscalemethod != "kurtosis"
		@error "Unsupported scaling method"
		error("Unsupported scaling method")
	end
    @debug "Using $(selfscale)"
    _, _, _, imgl = glg(img, sigmas); # Smoothing kernels, 3 is rough, 7 is fine
    Tg, neg_glog = find_th(imgl, z, selfscale, PRC, edgemask); # Z score, lower is more candidates, higher is fewer candidates
    # maxfilteredGMX, _ = processimg(img, neg_glog, Tg, 100, imgl);
    ngl = rlap(imgl) # neg 2nd
    i2 = copy(img)
    i2[ngl .< Tg] .= zero(eltype(img))
    i2[ngl .>= Tg] .= oneunit(eltype(img))
    # Get rid of artifacts on the border
    i2[1:3,:].=0
    i2[:,1:3].=0
    i2[end-2:end,:].=0
    i2[:,end-2:end].=0
    if smooth != 0
        @info "Smoothing"
        i2 = ImageFiltering.imfilter(i2, ImageFiltering.Kernel.gaussian(smooth))
    end
    ccs = Images.label_components(i2, trues(3,3))
    return ccs, imgl, Tg, img, i2
end

function processimg(img, neg_glog, Tg, TG, imgl)
    sx, sy = size(img)
    lmx = filterth(img, 0)
    maxfilteredGMX = [i for i in neg_glog if (-TG < imgl[i] < -Tg) & check(i[1], 1, sy, 5) & check(i[2], 1, sx, 5)];
    @info "Reduced $(size(neg_glog)) to $(size(maxfilteredGMX))"
    return maxfilteredGMX, nothing
end


"""
	Return the minimum distances for each mito component to the nearest C12 component, and the mask of corresponding C12 spots
	Return nothing, 0 if no mito components, Inf if no C12 component
"""
function quantify_adjacent_c12(mitomask, c12mask)
	#Return, for each mitocomp, the distance to the closest C12 spot
	c1c2_coms = Images.label_components(c12mask, trues(3,3))
	mito_coms = Images.label_components(mitomask, trues(3,3))
	NC12, NM = maximum(c1c2_coms), maximum(mito_coms)
	@debug "Have $NM mitochondria spots and $NC12 C12 spots"
	if NM == 0
		@debug "No mito spots"
		return nothing, ERGO.aszero(mitomask)
	end
	if NC12 == 0
		@debug "No C12 Spots"
		return [Inf64 for _ in 1:NM], ERGO.aszero(mitomask)
	end
	targetmask = ERGO.aszero(mitomask)
	nearestc12 = ERGO.aszero(mitomask)
	c12indices = Images.component_indices(c1c2_coms)[2:end]
	mindistances = zeros(NM)
	for (nth, indexn) in enumerate(Images.component_indices(mito_coms)[2:end])
		targetmask[indexn] .= 1
		distance_map = Images.distance_transform(Images.feature_transform(Bool.(targetmask)))
		minc12distances = [minimum(distance_map[ind]) for ind in c12indices]
		mind, minindex = findmin(minc12distances)
		nearestc12[c12indices[minindex]] .= 1
		mindistances[nth] = mind
		targetmask .= 0
	end
	return mindistances, nearestc12
end


"""
	tcolors(imagelist, transformfunctor)
	Return a colorview of up to 3 images where any missing are replaced with a zero image.
	Transformation is applied before colorview is returned, default identiy
"""
function tcolors(im2, transform=x -> x)
    N = length(im2)
	imgs = [transform(i) for i in im2]
    @match N begin
        3 => return Images.colorview(Images.RGB, imgs[1], imgs[2], imgs[3])
		2 => return Images.colorview(Images.RGB, imgs[1], imgs[2], ERGO.aszero(imgs[1]))
		1 => return Images.colorview(Images.RGB, imgs[1], ERGO.aszero(imgs[1]), ERGO.aszero(imgs[1]))
        _ => @assert(false)
    end
end


function record_stats(areas; clabel, metadata, distances, channelstats, c12_m_adjacencies, mindistances, intensities, QNM, cellwisestats, emptyspots, emptyresults, mito_under_c12m, cellimage, zeromito=false)
	# Needs a refactor
	# Use dicts with named columns, no positional :cry:
	serie = metadata["serie"]
	experiment = metadata["experiment"]
	celln = metadata["celln"]
	spots = copy(emptyspots)
	result = copy(emptyresults)
	if zeromito
		@debug "Mito = 0"
	end
	for (i,a) in enumerate(areas)  ### If i == C12 (3) --> then lookup vector of adjacencies and add adjacent_mito_spot_count=Int64[], mean_intensity_adjacent_mito=Float64[], std_intensity_adjacent_mito=Float64[]
		@debug "Processing channel $(clabel[i])"
		if i == 3 #C12
			μx, σx = channelstats[0]
			if length(a) > 0
				if ! zeromito
					@assert size(c12_m_adjacencies, 1) == length(a)
					@assert size(c12_m_adjacencies, 1) == size(mito_under_c12m, 1)
				end
			end
			for (_qi,_a) in enumerate(a)
				if !zeromito
					push!(spots, [celln, string(serie), experiment, clabel[i], _a, distances[_qi], c12_m_adjacencies[_qi, 1], c12_m_adjacencies[_qi, 2],
					c12_m_adjacencies[_qi, 3], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, QNM[_qi, 2], μx, QNM[_qi, 3], mito_under_c12m[_qi,1], mito_under_c12m[_qi,2], mito_under_c12m[_qi,3]])
				else
					push!(spots, [celln, string(serie), experiment, clabel[i], _a, Inf, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0])
				end
            end
			continue
		end
		if i == 7 #MITO
			if !zeromito
				μx, σx = channelstats[0]
				for (_qi,_a) in enumerate(a)
	                push!(spots, [celln, string(serie), experiment, clabel[i], _a, 0.0, 0.0, 0.0, 0.0, mindistances[_qi], intensities[1][_qi,1],
					intensities[1][_qi,2], intensities[1][_qi,3] , μx, σx, 0.0, μx, 0, 0.0, 0.0, 0.0])
	            end
			end
			continue
		end
		if i < 3 # C1, C2
			μx, σx = channelstats[i]
			for (_qi,_a) in enumerate(a)
				push!(spots, [celln, string(serie), experiment, clabel[i], _a, 0.0, 0.0, 0.0, 0.0, 0.0, intensities[i+1][_qi,1], intensities[i+1][_qi,2],
				intensities[i+1][_qi,3] , μx, σx, 0.0, 0.0, 0, 0.0, 0.0, 0.0])
			end
			continue
		end
		# ELSE : C1M, C2M, C12M
		if !zeromito
			for _a in a
					push!(spots, [celln, string(serie), experiment, clabel[i], _a, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0])
			end
		end
    end
	c1count, c2count, c1c2count, count_overlap_mito, count_overlap_c1_mito, count_overlap_c2_mito = cellwisestats

	push!(result, [celln, string(serie), experiment, c1count, c2count, c1c2count, count_overlap_mito, count_overlap_c1_mito, count_overlap_c2_mito, metadata[:area], metadata[:anisotropy], metadata[:λ1], metadata[:λ2]])
	return spots, result
end


"""
	Quantify the parameter-based adjacent mitophagy spots near C1C2
	Returns array where for each component i in C1C2, row i lists:
		the nr of spots, mean, and std of their intensities, nr of non zero pixels
	Second return value is the intensity mask
"""
function quantify_adjacent_mitophagy(c1c2_img, mito_img, raw_mito_img, TH_1, TH_2)
	@assert 0 <= TH_1 <= TH_2
	@assert size(c1c2_img) == size(mito_img)
	c1c2_coms = Images.label_components(ERGO.tomask(c1c2_img), trues(3,3))
	N = maximum(c1c2_coms)
	@debug "C1C2 $N components"
	if N == 0
		@debug "No C1C2 components"
		return nothing, ERGO.aszero(mito_img)
	end
	mito_coms = Images.label_components(ERGO.tomask(mito_img), trues(3,3))
	mito_intensity = raw_mito_img .* ERGO.tomask(mito_img)
	mint = mito_intensity[:]
	mint = Float64.(mint[mint .> 0])
	μi, σi = Statistics.mean(mint), Statistics.std(mint)
	@debug "Have $(maximum(mito_coms)) mitochondria spots"
	N = maximum(c1c2_coms)
	results = zeros(N, 6)
	resmask = ERGO.aszero(mito_img)
	q = nothing
	for (nth, indexn) in enumerate(Images.component_indices(c1c2_coms)[2:end])
		@debug "Checking $nth"
		# Nr of pixels 5-10, mean intensity, std, intensity
		cm = ERGO.aszero(c1c2_img)
		cm[indexn] .= 1
		distance_map = Images.distance_transform(Images.feature_transform(Bool.(cm)))
		## Get all the mito spots with a minimum distance < TH_1
		selected_mito = [ind for ind in Images.component_indices(mito_coms)[2:end] if minimum(distance_map[ind]) < TH_1]
		## Next, ensure we only look at slices of those spots at max distance < TH_2
		nd = copy(distance_map)
		nd[distance_map .>= TH_2] .= 0
		nd[distance_map .< TH_2] .= 1
		dm = ERGO.tomask(nd)
		sliced_mito = mito_intensity .* dm
		for mito_ind in selected_mito
			resmask[mito_ind] .= sliced_mito[mito_ind]
			if sum(resmask[mito_ind]) == 0
				@warn "Warning, residual mito mask can be corrupt!!"
			end
		end
		## Recompute
		adjacentmitointensities = [mito_intensity[ind].*dm[ind] for ind in selected_mito]
		nspots = length(adjacentmitointensities)
		if nspots > 0
			_i = Iterators.flatten(adjacentmitointensities) |> collect
			@debug "Total intensity == $(length(_i)) for $nspots"
			_ni = _i[_i .> 0]
			nz = length(_ni)
			μ, σ = Statistics.mean(Float64.(_ni)), Statistics.std(Float64.(_ni))
			if iszero(μ)
				@assert false
			end
			@debug "info $(nspots) with μ $(μ) ± $(σ)"
		else
			μ, σ, nz = 0.0, 0.0, 0.0
		end
		results[nth, :] .= nspots, μ, σ, nz, μi, σi
	end
	return results, resmask
end


"""
	Given two images (left, right), find the pixels in 'right' that are closer than k pixels to the nearest component in left.
	Report the mean, std intensities of those areas, and the distance map (binary) of the area around left
"""
function computeintensityboundary(left, right, k=2)
	coms = Images.label_components(ERGO.tomask(left), trues(3,3))
	ind = Images.component_indices(coms)[2:end]
	mc = ERGO.aszero(ERGO.tomask(left))
	N = maximum(coms)
	results = zeros(N, 2)
	for (nth, indexn) in enumerate(Images.component_indices(coms)[2:end])
		mc[indexn] .= 1
		dismap = Images.distance_transform(Images.feature_transform(Bool.(mc)))
		cd = copy(dismap)
		cd[dismap .< k] .= 1
		cd[dismap .>= k] .= 0
		msk = right .* cd
		m, s = Statistics.mean(msk[msk .> 0]), Statistics.std(msk[msk .> 0])
		@info "Component $nth has μ $m ± $s"
		results[nth, :] = [m, s ]
		mc .= 0
	end
	dismap = Images.distance_transform(Images.feature_transform(Bool.(ERGO.tomask(left))))
	dp = copy(dismap)
	dp[dismap .< k] .= 1
	dp[dismap .>= k] .= 0
	return results, dp
end


"""
	Compute distance for each connected component to a mask
	Returns a a 1XN vector of >= 0 where dis[5] is equal to smallest distance between pixels of component 5 and the mask
"""
function compute_distances_cc_to_mask(from_cc, to_mask)
    @assert(size(from_cc) == size(to_mask))
    dismap = Images.distance_transform(Images.feature_transform(Bool.(to_mask)))
    N = maximum(from_cc)
    dis = zeros(maximum(from_cc))
    ind = Images.component_indices(from_cc)[2:end]
    for i in 1:N
        @inbounds dis[i] = minimum(dismap[ind[i]])
    end
    @assert(all(dis .>= 0))
	@debug "Distances $N"
    return dis
end

function quantify_spots_mitophagy_new(res)
    ccmito, mask_mito = res[0]
    cc1, mask_1 = res[1]
    cc2, mask_2 = res[2]
    c1count = maximum(cc1)
    c2count = maximum(cc2)
    c1_c2 = copy(mask_1 .* mask_2)
    c1_c2_coms = Images.label_components(c1_c2, trues(3,3))
    c1c2count = maximum(c1_c2_coms)

    _fum = unionmask(mask_1, mask_2) # Save these
    c1_c2_u_comps = Images.label_components(_fum, trues(3,3))
    ctrs_mito = Images.component_centroids(ccmito)[2:end]
    ctrs_c1c2 = Images.component_centroids(c1_c2_u_comps)[2:end]
    # ds, ids = matchcentroids(c1_c2_u_comps, ccmito)
	if iszero(mask_mito)
		distances = []
	else
		distances = compute_distances_cc_to_mask(c1_c2_u_comps, ERGO.tomask(ccmito))
	end

    _c1c2m = _fum .* mask_mito
    overlap_mito_coms = Images.label_components(_fum .* mask_mito, trues(3,3))
    c1c2mitocount = maximum(overlap_mito_coms)

    overlap_c1_mito_coms = Images.label_components(mask_1 .* mask_mito, trues(3,3))
    overlap_c2_mito_coms = Images.label_components(mask_2 .* mask_mito, trues(3,3))
    count_overlap_c1_mito = maximum(overlap_c1_mito_coms)
    count_overlap_c2_mito = maximum(overlap_c2_mito_coms)
    _ccs = [ cc1, cc2, c1_c2_u_comps, overlap_mito_coms, overlap_c1_mito_coms, overlap_c2_mito_coms, ccmito ]
    areas = [Images.component_lengths(_cc)[2:end] for _cc in _ccs]
	images = res[3]
	im_mito = images[0] .* mask_mito
	im_1 = images[1] .* mask_1
	im_2 = images[2] .* mask_2
	masked_images = [im_mito, im_1, im_2]
	intensities = []
	channelintensities = Dict()
	@debug "Processing channels"
	for (ic, channel) in enumerate([ccmito, cc1, cc2])
		@debug "Channel $ic"
		channel_image = masked_images[ic]
		N = maximum(channel)

		indices = Images.component_indices(channel)[2:end]
		INT = zeros(Float64, N, 3)
		if iszero(channel)
			@debug "Zero image, no stats to compute"
			push!(intensities, INT)
			channelintensities[ic-1] = 0, 0
		else
			nzimg = Float64.(channel_image[channel_image .> 0])
			statchannel = describeimg(nzimg)
			channelintensities[ic-1] = statchannel[1], statchannel[3]
			for (ith, indexes_ith) in enumerate(indices)
				int_comp = channel_image[indexes_ith][:]
				descr = describeimg(int_comp)
				cd = cohen_d(int_comp, nzimg[:])
				INT[ith, :] .= [descr[1], descr[3], cd]
			end
			push!(intensities, INT)
		end
	end
    return c1count, c2count, c1c2count, c1c2mitocount, count_overlap_c1_mito, count_overlap_c2_mito, areas, distances, _fum, intensities, channelintensities
end

"""
	If channels are not cleanly organized, try to recover and sort the tiffs. Assumes ....{0,1,2}.tif as pattern.
"""
function orderchannels(tiffs)
	res = Dict()
	for tif in tiffs
		channelnr = split(basename(tif), ".")[end-1][end]
		cnr = tryparse(Int, "$channelnr")
		@assert 0 <= cnr < 3
		res[cnr] = tif
	end
	cnrs = []
	orderedtiffs = []
	for k in sort(keys(res)|>collect)
		push!(cnrs, k)
		push!(orderedtiffs, res[k])
	end
	return cnrs, orderedtiffs
end

#
# """
# 	score_masks_visual(gt_mask, pr_mask)
# 	Returns  fp, tp, fn where a match is counted if >= 1 pixel overlaps.
# 	Input and output are binary masks.
# """
# function score_masks_visual(gt_mask, pr_mask)
# 	fp, tp, fn = ERGO.aszero(gt_mask), ERGO.aszero(gt_mask), ERGO.aszero(gt_mask)
# 	gt_ccs = Images.label_components(gt_mask)
# 	pr_ccs = Images.label_components(pr_mask)
# 	gt_inds = Images.component_indices(gt_ccs)[2:end]
# 	pr_inds = Images.component_indices(pr_ccs)[2:end]
# 	for gt_i in gt_inds
# 		if any(pr_mask[gt_i] .> 0)
# 			tp[gt_i] .= 1
# 		else
# 			# Not covered
# 			fn[gt_i] .= 1
# 		end
# 	end
# 	for pr_i in pr_inds
# 		if sum(gt_mask[pr_i]) == 0
# 			fp[pr_i] .= 1
# 		end
# 	end
# 	return fp, tp, fn
# end


"""
	score_masks_mcc(gt_mask, pr_mask)
	Compute Matthews Correlation coefficient on GT, predicted (image masks)
"""
function score_masks_mcc(gt_mask, pr_mask)
	fp, tp, fn = 0, 0, 0
	gt_ccs = Images.label_components(gt_mask)
	pr_ccs = Images.label_components(pr_mask)
	gt_inds = Images.component_indices(gt_ccs)[2:end]
	pr_inds = Images.component_indices(pr_ccs)[2:end]
	for gt_i in gt_inds
		if any(pr_mask[gt_i] .> 0)
			tp += 1
		else
			# Not covered
			fn += 1
		end
	end
	for pr_i in pr_inds
		if sum(gt_mask[pr_i]) == 0
			fp += 1
		end
	end
	return fp, tp, fn, 0, ERGO.mcc(tp, 0, fp, fn)
end

"""
	Try to load and validate that in path there are 3 equally sized images, *[0,1,2].tif
	If *0.tif is missing, don't fail but prefix a zeroed copy of *1.tif
	Any error results in returning nothing, else an array of images sorted by file name
"""
function validatechannels(path)
	tiffs = Glob.glob("*[0,1,2].tif", path)
	channelnrs, channels = orderchannels(tiffs)
	images=[]
	try
		images = [Images.load(tif) for tif in channels]
	catch err
		@error err
		return nothing
	end
	dims = [size(img) for img in images]
	fst = dims[1]
	for d in dims[2:end]
		if length(d) != 2
			@warn "Invalid image dimensions, aborting"
			return nothing
		end
		if d != fst
			@warn "Invalid image dimensions, aborting"
			return nothing
		end
	end
	types = [eltype(i) for i in images]
	if ! all(x->x==types[1], types)
		@warn "Invalid image types $(types), aborting"
		return nothing
	end
	for t in types
		if !(t == Images.Gray{Images.N0f8}) | (t == Images.Gray{Images.N0f16})
			@warn "Invalid image types $(t), aborting"
			return nothing
		end
	end
	Nimg = length(images)
	if Nimg == 3
		@debug "Validated images ✓"
		return images
	end
	if (Nimg == 2) & (channelnrs[1] == 1)
		@debug "No mito channel, adding zeroed channel"
		# mito = [ERGO.aszero(images[1])]
		return [ERGO.aszero(images[1]), images[1], images[2]]
		# return vcat([mito, images...])
	end
	@warn "Invalid number of images $(channelnrs) != 2,3"
	return nothing
end

"""
	Parse the 3 channels of a cell
	:channels : list of integers, matching files of the patterin *ch0[channel].tif in qpath
"""
function process_cell(qpath, channels, outdir, serie, subdir, experiment, z, selfscale, celln, SQR; maxspotsize=Inf64, mode="default",
	sigma=3.0, edgeiters=1, PRC=3.75, quantile=0.9)
	images = validatechannels(qpath)
	if isnothing(images)
		 return nothing, 3, true, nothing
	end
	@debug "Images = $(length(images))"
	res = Dict()
	resulting_images = Dict()
	### Use match "mode" --> function
	### Add mode : snip out cells
	if mode =="apery"
		@debug "APERY"
		results = process_non_segmented_non_decon_channels(images; PRC=PRC, sigma=sigma, quantile=quantile, pixprec=SQR)
		if isnothing(results)
			@debug "Processing failed for $(experiment) Serie $(serie) Cell $(celln) due to segmentation error"
			return nothing, 3, true, nothing
		end
		for channelkey in keys(results)
			maskedcell, cmsk, outline, edgemask, cimg, channelcellmask = results[channelkey]
			# SEGMENTED, FT, outline, em, cimg, segment = res[2]
			cmsk = filtercomponents_using_maxima_IQR(maskedcell, cmsk)
			outline = ♌(cmsk)
			Images.save(joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_channel_$(channelkey)_cell_mask.tif"), Images.Gray{Images.N0f16}.(channelcellmask))
			Images.save(joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_channel_$(channelkey)_cell_edge_mask.tif"), Images.Gray{Images.N0f16}.(edgemask))
			Images.save(joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_channel_$(channelkey)_cell_object_outline.tif"), Images.Gray{Images.N0f16}.(outline))
			ccs = Images.label_components(cmsk, trues(3,3))
			res[channelkey] = (ccs, cmsk)
			resulting_images[channelkey] = maskedcell
			outf = joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_channel_$(channelkey)_mask_filtered_$(SQR).tif")
			Images.save(outf, Images.Gray{Images.N0f16}.(cmsk))
		end
		@debug "Apery mode done, returning"
		res[5] = Dict(0=>images[1], 1=>images[2], 2=>images[3])
		return res, 0, false, resulting_images
	end
	# if mode == "segment"
		# Do segment, then default
	# if mode == default do default
	cellmasks = []
	for (channelrev, image) in enumerate(reverse(images))
		channel = 3 - channelrev # 3->0, 2->1, 1->2
		edgemask = nothing
		if mode == "segment"
			@debug "deprecated"
			if channel == 0
				@assert length(cellmasks) == 2
				image = image .* (cellmasks[1][1] .+ cellmasks[2][1])
				cellmask = ERGO.tomask(image)
				edgemask = ERGO.tomask(cellmasks[1][2] .+ cellmasks[2][2])
				Images.save(joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_channel_$(channel)_cell_mask.tif"), cellmask)
				Images.save(joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_channel_$(channel)_cell_edge_mask.tif"), edgemask)
				resulting_images[channel] = image
			else
				if iszero(image)
					@warn "Refusing to segment zero image for channel $(channel)"
					resulting_images[channel] = image
					push!(cellmasks, (ERGO.aszero(image), ERGO.aszero(image)))
				else
					@debug "Segmenting $channel "
					segmented_image, cellmask, edgemask = lowsnrsegment(image, edgeiters=edgeiters)
					segratio = (sum(cellmask) / sum(ERGO.tomask(image)))
					if segratio > .99
						@debug "Segmentation probably failed for channel $channel"
					end
					@debug "Segmentation reduce to $(sum(cellmask) / sum(ERGO.tomask(image)))"
					Images.save(joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_channel_$(channel)_cell_mask.tif"), cellmask)
					Images.save(joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_channel_$(channel)_cell_edge_mask.tif"), edgemask)
					image = segmented_image
					push!(cellmasks, (cellmask, edgemask))
					resulting_images[channel] = image
				end
			end
		else
			@debug "No segmentation"
			resulting_images[channel] = image
			res[5] = Dict(0=>images[1], 1=>images[2], 2=>images[3])
		end
		### Replace by function
		# if skipmito save zeros, zeros
		if iszero(image)
			@debug "Mito image zeroed, skipping processing"
			res[channel] = (Images.label_components(image), image)
		else
			ccs, imgl, Tg, _img, msk = process_tiffimage(image, z, [sigma, sigma], selfscale, PRC, 0, edgemask=edgemask)
			if selfscale
				@warn "Prototype filtering"
				R = denoisemd(image, 3)
				R[R .< 0] .= 0
				R = Images.N0f8.(R)
				mask = ERGO.tomask(iterative(R, ImageMorphology.dilate, SQR))
				edge = ⨣(𝛁(mask))
				edged = ERGO.tomask(iterative(edge, ImageMorphology.dilate, SQR))
				SF = dropartifacts(ccs, image)
				ccs = Images.label_components(ERGO.tomask(SF), trues(3,3))
			end
			outf = joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_channel_$(channel)_mask.tif")
			Images.save(outf, msk)
			cmsk = filter_cc_sqr_greater_than(ccs, _img, SQR)
			ccs = Images.label_components(cmsk, trues(3,3))
			outf = joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_channel_$(channel)_mask_filtered_$(SQR).tif")
			Images.save(outf, cmsk)
			res[channel] = (ccs, cmsk)
		end
	end
	# return default_specht_spots([resulting_images[k] for sort(keys(resulting_images)|>collect)])
	### If mode == default
	### return default_specht_spots
	return res, 0, false, resulting_images
end

function default_specht_spots(images; z, sigma, selfscale, prc, edgemasks=nothing, SQR, metadata)
	### TODO integrate
	for (ichannel, image) in enumerate(images)
		channel = ichannel - 1
		ccs, imgl, Tg, _img, msk = process_tiffimage(image, z, [sigma, sigma], selfscale, prc, 0, edgemask= isnothing(edgemasks) ? edgemasks : edgemasks[ichannel] )
		outf = joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_channel_$(channel)_mask.tif")
		Images.save(outf, msk)
		cmsk = filter_cc_sqr_greater_than(ccs, _img, SQR)
		ccs = Images.label_components(cmsk, trues(3,3))
		outf = joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_channel_$(channel)_mask_filtered_$(SQR).tif")
		Images.save(outf, cmsk)
		res[channel] = (ccs, cmsk)
	end
	return res, 0, false, Dict((i-1)=>images[i] for i in 1:length(images))
end


function computemasks(res, union_mask_c12)
	_c1mask = res[1][end]
	_c2mask = res[2][end]
	_mitomask = res[0][end]
	_c1prime = ERGO.tomask(Int8.(_c1mask) .+ union_mask_c12)
	_c2prime = ERGO.tomask(Int8.(_c2mask) .+ union_mask_c12)
	tricolor_result_no_union = Images.colorview(Images.RGB, _c1mask, _c2mask, _mitomask)
	tricolor_result_union = Images.colorview(Images.RGB, _c1prime, _c2prime, _mitomask)
	return tricolor_result_no_union, tricolor_result_union
end

"""
	Similar to find_th, but applied to intensities.
	dropzeros : use geometric mean (hence zeros are ignored)
	autotune : use kurtosis to find k
	PRC : scale kurtosis based k
"""
function filter_k(img, k, dropzeros=false, autotune=false, PRC=1.0)
	z = zero(eltype(img))
	_img = copy(img)
	if autotune
		@debug "Using autotuning"
		@debug "Ignoring $k to ..."
		k = scale(_img)
		if k == 0.0
			@error "Kurtosis near zero, meaningless results will follow, dropping entire image, SNR too low"
			_img .= z
			return _img, typemax(eltype(img))
		end
		@debug "... $k to ..."
		k = k/PRC
		@debug "... $(k) with PRC $(PRC)"
	end
	if dropzeros
		# μ, σ = mean(_img[_img .> z]), std(_img[_img .> 0])
		@debug "Using geometric mean"
		μ, σ = ERGO.gmsm(_img)
		@debug "μ $μ σ $σ "
		th = μ * σ^k
	else
		@debug "Using arithmetic mean"
    	μ, σ = Statistics.mean(Float64.(_img)), Statistics.std(Float64.(_img))
		th = μ + σ*k
	end
    _img[_img .< th] .= z
    return _img,  th
end

"""
	Delegate function, process spots for mitophagy analysis
"""
function qspot(res, oldbehavior)
    if oldbehavior == true
        return quantify_spots_mitophagy(res)
    else
        return quantify_spots_mitophagy_new(res)
    end
end

"""
	Harmonic mean of array of >= 0 values, zeros are ignored
"""
function harmonicmean(vals)
	z = zero(eltype(vals))
	@assert(all(vals .>= z))
	vs = vals[vals .> z]
	S = sum(1 ./ vs)
	return length(vs)/S
end


"""
	Union of 2 masks.
"""
function unionmask(mask_1, mask_2)
    _c1_c2_union = Int8.(mask_1) + Int8.(mask_2)  # a ConCom will have a 3 if it shares an intersection
    _c1_c2_u = ERGO.tomask(_c1_c2_union)
    _c1_c2_c = Images.label_components(_c1_c2_u, trues(3,3))
    _ind = Images.component_indices(_c1_c2_c)[2:end]
    _filtered_u = ERGO.aszero(_c1_c2_union)
    for _cc in 1:maximum(_c1_c2_c)
        _indx = _ind[_cc]
        if maximum(_c1_c2_union[_indx]) == 2
            _filtered_u[_indx] .= 1
        end
    end
    return ERGO.tomask(_filtered_u)
end

"""
	Compute Guassian - Laplacian - Guassian transform of image.
	σ1, σ2 determine the first, second Guassian.
"""
function glg(img, sigmas)
    @assert(length(sigmas)==2)
    imgg = img
    if sigmas[1] != 0
        imgg = ImageFiltering.imfilter(img, ImageFiltering.Kernel.gaussian(sigmas[1]));
    end
    # imgg = img
    imgl = ImageFiltering.imfilter(imgg, ImageFiltering.Kernel.Laplacian());
    imglg = imgl
    if sigmas[2] != 0
        imglg = ImageFiltering.imfilter(imgl, ImageFiltering.Kernel.gaussian(sigmas[2]));
    end
    return img, imgg, imgl, imglg
end

"""
	Compute the negative laplacian of 2 images
"""
function compute_nl(s1, s2)
    # ll1 = ImageFiltering.imfilter(s1, ImageFiltering.Kernel.Laplacian());
    # ll2 = ImageFiltering.imfilter(s2, ImageFiltering.Kernel.Laplacian());
    # ln1 = copy(ll1)
    # ln2 = copy(ll2)
	# zr = zero(eltype(s1))
    # ll1[ll1 .> zr] .= zr
    # ll2[ll2 .> zr] .= zr
    # ln1[ln1 .< zr] .= zr
    # ln2[ln2 .< zr] .= zr
    # return ll1, ll2, ln1, ln2
	ll1, ln1 = cnl(s1)
	ll2, ln2 = cnl(s2)
	return ll1, ll2, ln1, ln2
end

function cnl(img)
	ll1 = ImageFiltering.imfilter(img, ImageFiltering.Kernel.Laplacian());
	ln1 = rlap(ll1)
	return ll1, ln1
end

end # module
