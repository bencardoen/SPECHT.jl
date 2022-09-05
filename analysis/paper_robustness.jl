using Pkg
Pkg.activate(".")
using Images, SPECHT, Plots
using ImageView
using Statistics
using ERGO
using Glob

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


### Load 1 MC5 cell, unannotated, and compare

σ = 1.25
PRC = 4.25


## MC5
mc5p = "/home/bcardoen/storage/cedar_data/specht/SPECHT_DATASETS/datasets/Cav1/data/Cav1R1/MC5_Decon/Series003_decon_converted"
tfs=sort(Glob.glob("*.tif", mc5p))
ims = [Images.load(t) for t in tfs]
imshow(ims[2])

M=ims[2]
mccs, mll = SPECHT.process_tiffimage(M, 0, [0,σ], true, PRC, 0)[1:2];
fts_mc5 = getfeatures(M, mccs, rlap(mll))

#
# function analyzeimage(image, σ, PRC)
# 	_ccs, mll = SPECHT.process_tiffimage(image./maximum(image), 0, [0,σ], true, PRC, 0)[1:2];
# 	fts = getfeatures(image, _ccs, rlap(mll))
# 	return _ccs, fts
# end



## Cav1 in PC3PTRF
# path = "/home/bcardoen/storage/cedar_data/tim/annotationed cav1 trial"
path = "/home/bcardoen/storage/cedar_data/tim/T2/Series004_decon_converted/"
output = "/home/bcardoen/storage/cedar_data/tim/T2/Series004_decon_converted/output20220814"
mkpath(output)

AN="/home/bcardoen/storage/cedar_data/tim/T2/annoatated_Cav1.tif"
IN="/home/bcardoen/storage/cedar_data/tim/T2/Series004_decon_converted_ch01.tif"

A=Images.load(AN)
# imshow(A)
I=Images.load(IN)
# imshow(SPECHT.tcolors([A, I]))


#Process PC3PTRF
# ccs, cll = SPECHT.process_tiffimage(I, 0, [0,σ], true, PRC, 0)[1:2];
# fts_pc3 = getfeatures(I, ccs, rlap(cll))
mccs, fts_mc5 = analyzeimage(M, σ, PRC)
ccs, fts_pc3 = analyzeimage(I, σ, PRC)
#
# scatter(fts_pc3[:,1], fts_pc3[:,3])
# scatter!(fts_mc5[:,1], fts_mc5[:,3])

pc_to_5 = contrast_x_to_y(fts_pc3[:,[2]], fts_mc5[:,[2]])
pc_to_c = contrast_x_to_y(fts_pc3[:,[2]], fts_pc3[:,[2]])
labeled_5 = label_image(ccs, max.(0.25, pc_to_5))
labeled_c = label_image(ccs, max.(0.25, 1 .- pc_to_c))
imshow(maskoutline(labeled_c))

scatter(1 .- pc_to_c, fts_pc3[:,2])
scatter!(pc_to_5, fts_pc3[:,2])


imshow(SPECHT.tcolors([ fuse_images(GTI, maskoutline(labeled_5)), fuse_images(GTI, maskoutline(labeled_c)), GTI .+ ERGO.tomask(A)]))

### Find the ROIs
rois, roibox = to_rois(ERGO.tomask(A), 300)
Images.save(joinpath(output, "inset.tif"), max.(ERGO.tomask(rois), I))
roiI, roiA, roiS, roi5, roiC = [slicerois(i, roibox) for i in [I, ERGO.tomask(A), ERGO.tomask(labeled_5 .+ labeled_c), labeled_5, labeled_c]]

## Save GT
for RI in 1:length(roiS)
	_i = roiI[RI]
	Images.save(joinpath(output, "annotate_roi_$(RI)_gt.tif"), _i)
end

# nm = i -> i ./ maximum(i)
fm = (x, y) -> fuse_max(x, y)

# _i = SPECHT.tcolors( [ fm(GTRI,maskoutline(roi5[RI])), fm(GTRI, maskoutline(roiC[RI])), GTRI])

for RI in 1:length(roiS)
	# fuse_images(GTI, maskoutline(labeled_5))
	GTRI = fm(roiI[RI], roiA[RI])
	_i=SPECHT.tcolors( [ fm(GTRI,maskoutline(roi5[RI])), fm(GTRI, maskoutline(roiC[RI])), GTRI])
	Images.save(joinpath(output, "annotate_roi_$(RI).tif"), _i)
end




## Degrade with noise
X, Y = size(I)
## Compute for PC3
	## TODO same for MC5
	## Compute features and labels
	## Save= size(I)
facs = [16]
## Reuses the 'total' image from previous snippet
results  = Dict()
pc3 = copy(I)
mc5 = copy(M)
j=1
function addnoise(_img, fc)
	X, Y = size(_img)
	gns = SPECHT.gaussiannoise(zeros(X, Y), fc)
	pns = SPECHT.poissonnoise(zeros(X, Y), fc)
	clamp01.(gns .+ pns .+ _img)
end
r=gm(pc3)/gm(mc5)

pc3n = addnoise(pc3./maximum(pc3), 8)
_ccs, mll = SPECHT.process_tiffimage(pc3n./maximum(pc3n), 0, [0,σ], true, PRC, 0)[1:2];
cmsk = filtercomponents_using_maxima_IQR(pc3n, ERGO.tomask(_ccs))
nccs=label_components(cmsk)
fts = getfeatures(pc3n, nccs, rlap(mll))

mc5n = addnoise(mc5./maximum(mc5), 8)
_ccs, mll = SPECHT.process_tiffimage(mc5n./maximum(mc5n), 0, [0,σ], true, PRC, 0)[1:2];
cmsk = filtercomponents_using_maxima_IQR(mc5n, ERGO.tomask(_ccs))
n5ccs=label_components(_ccs)
fts5 = getfeatures(mc5n, n5ccs, rlap(mll))



pcn_to_5 = contrast_x_to_y(fts[:,[2]], fts5[:,[2]])
pcn_to_c = contrast_x_to_y(fts[:,[2]], fts[:,[2]])
labeled_5n = label_image(nccs, max.(0.25, pcn_to_5))
labeled_cn = label_image(nccs, max.(0.25, 1 .- pcn_to_c))



GTIN = fuse_max(pc3n, ERGO.tomask(A))

imshow(SPECHT.tcolors([ fuse_images(GTIN, maskoutline(labeled_5n)), fuse_images(GTIN, maskoutline(labeled_cn)), GTIN .+ ERGO.tomask(A)]))

### Find the ROIs
rois, roibox = to_rois(ERGO.tomask(A), 300)
Images.save(joinpath(output, "inset.tif"), max.(ERGO.tomask(rois), I))
roiIn, roiAn, roiSn, roi5n, roiCn = [slicerois(i, roibox) for i in [pc3n, ERGO.tomask(A), ERGO.tomask(labeled_5n .+ labeled_cn), labeled_5n, labeled_cn]]

## Save GT
for RI in 1:length(roiSn)
	_i = roiIn[RI]
	Images.save(joinpath(output, "annotate_roi_noise_$(RI)_gt.tif"), _i)
end

# nm = i -> i ./ maximum(i)
fm = (x, y) -> fuse_max(x, y)

# _i = SPECHT.tcolors( [ fm(GTRI,maskoutline(roi5[RI])), fm(GTRI, maskoutline(roiC[RI])), GTRI])

for RI in 1:length(roiSn)
	# fuse_images(GTI, maskoutline(labeled_5))
	GTRI = fm(roiIn[RI], roiAn[RI])
	_i=SPECHT.tcolors( [ fm(GTRI,maskoutline(roi5n[RI])), fm(GTRI, maskoutline(roiCn[RI])), GTRI])
	Images.save(joinpath(output, "annotate_roi_$(RI)_noise.tif"), _i)
end



	# roiI, roiA, roiS = [slicerois(i, roibox) for i in [noisedimage, A, out]]
# 	# RI=1
# 	for RI in 1:length(roiS)
# 		_i = SPECHT.tcolors( [ roiI[RI]./maximum(roiI[RI]), roiA[RI]./maximum(roiA[RI]), roiS[RI]./maximum(roiS[RI])])
# 		Images.save(joinpath(output, "annotate_roi_$(RI)_noise_$j.tif"), _i)
#
# 		_i = roiI[RI]
# 		Images.save(joinpath(output, "annotate_roi_$(RI)_noise_$(j)_gt.tif"), _i)
# 	end
# 	results[NSFAC] = ccs, out, noisedimage, cmsk, roiI, roiA, roiS
# end
#

ccs, out, ni, cmsks, roiI, roiA, roiS=results[facs[1]]

imshow(SPECHT.tcolors([ni, ERGO.tomask(A), SPECHT.maskoutline(ERGO.tomask(ccs))]))
imshow(SPECHT.tcolors([ni, ERGO.tomask(A), out]))



### Functions

function annotate_at(img, coords, vals)
    _im = copy(img)
    for (i,coord) in enumerate(coords)
        @info i
        @info coord
        @info vals[i]
        @info _im[coord...]
        _im[coord...] = vals[i]
    end
    _im
end

function annotate_cross_at(img, coords, vals, span=5)
    _im = copy(img)
    for (i,coord) in enumerate(coords)
        @info i
        @info coord
        @info vals[i]
        @info _im[coord...]
        x, y = coord
        # _im[coord...] = vals[i]
        _im[x-span: x+span, y] .= vals[i]
        _im[x, y-span:y+span] .= vals[i]
    end
    _im
end


function getfeatures(imgs, ccs, cll)
	N = maximum(ccs)
	F = zeros(N, 3)
	ls = Images.component_lengths(ccs)[2:end]
	si = Images.component_indices(ccs)[2:end]
	F[:,1] .= ls
	for i in 1:N
		nt = imgs[si[i]]
		nl = cll[si[i]]
		F[i,2] = sum(nt)
		F[i,3] = sum(nl)
	end
	return F
end

function integercentroids(ctrs)
    return intcent.(ctrs)
end

function intcent(c)
    return Int.(round.(c))
end

function cluster(ccs)
    ctrs = component_centroids(ccs)[2:end]
    N = length(ctrs)
    mt = zeros(N, N)
    X, Y = size(ccs)
    mt .= sqrt(X^2 + Y^2)
    for i in 1:N
        for j in 1:N
            dij = sqrt(sum((ctrs[i] .- ctrs[j]).^2))
            mt[i,j] = dij
            mt[j,i] = dij
        end
    end
    mt
end

function findrect(comp, cc)
    minx, miny, MAXX, MAXY = Inf, Inf, 0, 0
    for c in comp
        l, L = component_boxes(cc)[c+1]
        mx, my = l
        MX, MY = L
        minx = Int.(min(minx, mx))
        miny = Int.(min(miny, my))
        MAXX = Int.(max(MAXX, MX))
        MAXY = Int.(max(MAXY, MY))
    end
    return [[minx, miny], [MAXX, MAXY]]
end

function annotate_roi(img, coords, vals)
    _img = ERGO.aszero(img)
    for (i,coord) in enumerate(coords)
        mx, my = coord[1]
        MX, MY = coord[2]
        _img[mx:MX, my:MY] .= vals[i]
    end
    return _img
end

function torect(img)
    _img = copy(img)
    d=dilate(dilate(_img))
    e=erode(_img)
    return d .- e
end

function to_rois(annotation, distance=200)
	_ccs=label_components(ERGO.tomask(annotation))
	mt = cluster(_ccs)
	graph = buildGraph(mt, distance)
	N = maximum(_ccs)
	CC = connectedComponents(N, graph)
	rects = [findrect(c, _ccs) for c in CC]
	ROI=annotate_roi(ERGO.aszero(annotation), rects, [1 for r in rects])
	R2=torect(ROI)
	CRS = label_components(R2)
	roibox = component_boxes(CRS)[2:end]
	return CRS, roibox
end

function slicerois(im, boxes, alignto=160)
	res = []
	for box in boxes
		(xmin, ymin), (xmax, ymax) = box
		xr = xmax-xmin
		yr = ymax-ymin
		borderx = alignto - xr
		padx=true
		if borderx % 2 == 0
			padx = false
			bx = Int(borderx/2)
		else
			bx = Int((borderx-1)/2)
		end
		bordery = alignto - yr
		pady=true
		if bordery % 2 == 0
			pady = false
			by = Int(bordery/2)
		else
			by = Int((bordery-1)/2)
		end
		@info borderx
		@info bordery
		@info xmin ymin xmax ymax
		push!(res, copy(N0f16.(im[xmin-bx-(padx ? 1 : 0):xmax+bx, ymin-by-(pady ? 1 : 0) : ymax+by])))
	end
	res
end
function label_image(cx, px)
	res = Images.N0f16.(ERGO.aszero(cx))
	indices = Images.component_indices(cx)[2:end]
	for (j, ind) in enumerate(indices)
		res[ind] .= px[j]
	end
	res
end

function analyzeimage(image, σ, PRC)
	_ccs, mll = SPECHT.process_tiffimage(image./maximum(image), 0, [0,σ], true, PRC, 0)[1:2];
	fts = getfeatures(image, _ccs, rlap(mll))
	return _ccs, fts
end

function fuse_images(x, y)
	return N0f16.(clamp01.(Float64.(x) .+ Float64.(y)))
end



function fuse_max(x, y)
	return N0f16.(clamp01.(max.(Float64.(x) , Float64.(y))))
end
