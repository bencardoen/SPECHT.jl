
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

using Images, SPECHT, Plots
using ImageView
using Statistics
using ERGO
using Colocalization
using Glob

## Note on usage
## See README on datasets
## This file is used interactively, e.g. in Atom, not as a script.

## Supporting functions

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
		F[i,2] = mean(nt)
		F[i,3] = mean(nl)
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
    _img = Colocalization.aszero(img)
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
	_ccs=label_components(Colocalization.tomask(annotation))
	mt = cluster(_ccs)
	graph = buildGraph(mt, distance)
	N = maximum(_ccs)
	CC = connectedComponents(N, graph)
	rects = [findrect(c, _ccs) for c in CC]
	ROI=annotate_roi(Colocalization.aszero(annotation), rects, [1 for r in rects])
	R2=torect(ROI)
	CRS = label_components(R2)
	roibox = component_boxes(CRS)[2:end]
	return CRS, roibox
end

function slicerois(im, boxes, alignto=130)
	res = []
	for box in boxes
		(xmin, ymin), (xmax, ymax) = box
		xr = xmax-xmin
		yr = ymax-ymin
		borderx = alignto - xr
		bordery = alignto - yr
		@info borderx
		@info bordery
		@info xmin ymin xmax ymax
		push!(res, copy(N0f16.(im[xmin-1:xmax+borderx, ymin-1:ymax+bordery])))
	end
	res
end


function label_image(cx, px)
	res = Images.N0f16.(Colocalization.aszero(cx))
	indices = Images.component_indices(cx)[2:end]
	for (j, ind) in enumerate(indices)
		res[ind] .= px[j]
	end
	res
end


## MC5
mc5p = "/home/bcardoen/storage/cedar_data/specht/SPECHT_DATASETS/datasets/Cav1/data/Cav1R1/MC5_Decon/Series003_decon_converted"
tfs=sort(Glob.glob("*.tif", mc5p))
ims = [Images.load(t) for t in tfs]
imshow(ims[2])

M=ims[2]
σ = 1.25
mccs, mll = SPECHT.process_tiffimage(M, 0, [0,σ], true, 4.25, 0)[1:2];
mcmsk = filtercomponents_using_maxima_IQR(M, Colocalization.tomask(mccs))

fts_mc5 = getfeatures(M, mccs, mll)


path = "/home/bcardoen/storage/cedar_data/tim/T2/Series004_decon_converted/"
output = "/home/bcardoen/storage/cedar_data/tim/T2/Series004_decon_converted/output"
mkpath(output)


AN="/home/bcardoen/storage/cedar_data/tim/T2/annoatated_Cav1.tif"
IN="/home/bcardoen/storage/cedar_data/tim/T2/Series004_decon_converted_ch01.tif"

A=Images.load(AN)
I=Images.load(IN)



σ = 1.25
ccs, cll = SPECHT.process_tiffimage(I, 0, [0,σ], true, 4.25, 0)[1:2];
cmsk = filtercomponents_using_maxima_IQR(I, Colocalization.tomask(ccs))
SM = Colocalization.tomask(ccs)
OUT=SPECHT.maskoutline(SM)


fts_pc3 = getfeatures(I, ccs, cll)

pc_to_5 = contrast_x_to_y(fts_pc3, fts_mc5)
## For each x ∈ X, how likely is it to appear in X ?
p5_to_c = contrast_x_to_y(Fx[:,2:2], Fx[:,2:2])

cx = components[1]
r, g = [label_image(cx, p) for p in [px_to_y, px_to_x]]

imshow(SPECHT.tcolors([Colocalization.tomask(A), I, OUT]))
# imshow(SPECHT.tcolors([Colocalization.tomask(A), I, cmsk]))

### Find the ROIs

rois, roibox = to_rois(Colocalization.tomask(A), 300)
Images.save(joinpath(output, "inset.tif"), max.(Colocalization.tomask(rois), I))
roiI, roiA, roiS = [slicerois(i, roibox) for i in [I, A, OUT]]

## Save GT
for RI in 1:length(roiS)
	_i = roiI[RI]
	Images.save(joinpath(output, "annotate_roi_$(RI)_gt.tif"), _i)
	Images.save(joinpath(output, "annotate_roi_$(RI)_gts4.tif"), _i./4)
end

nm = i -> i ./ maximum(i)

for RI in 1:length(roiS)
	_i = SPECHT.tcolors( [ nm(roiI[RI]), nm(roiA[RI]), nm(roiS[RI])])
	Images.save(joinpath(output, "annotate_roi_$(RI).tif"), _i)
end



σ = 1.25
ccs = SPECHT.process_tiffimage(I./4, 0, [0,σ], true, 4.25, 0)[1];
cmsk = filtercomponents_using_maxima_IQR(I, Colocalization.tomask(ccs))
SM = Colocalization.tomask(ccs)
OUT=SPECHT.maskoutline(SM)

rois, roibox = to_rois(Colocalization.tomask(A), 300)
roiI, roiA, roiS = [slicerois(i, roibox) for i in [I, A, OUT]]

# RI=1
for RI in 1:length(roiS)
	_i = SPECHT.tcolors( [ roiI[RI], roiA[RI]./maximum(roiA[RI]), roiS[RI]./maximum(roiS[RI])])
	Images.save(joinpath(output, "annotate_roi_$(RI)_lowint.tif"), _i)
end


imshow(SPECHT.tcolors([Colocalization.tomask(A), I, OUT]))




## Degrade

X, Y = size(I)
facs = [8, 16, 32]
## Reuses the 'total' image from previous snippet
results  = Dict()
total = copy(I)
for (j, NSFAC) in enumerate(facs)
	@info "Noise factor $(NSFAC/255)"
	gns = SPECHT.gaussiannoise(zeros(X, Y), NSFAC)
	pns = SPECHT.poissonnoise(zeros(X, Y), NSFAC)
	totalnoise = gns .+ pns
	noisedimage = ERGO.normimg(totalnoise .+ total)
	ccs = SPECHT.process_tiffimage(noisedimage, 0, [σ,σ], true, 4.25, 0)[1];
	cmsk = filtercomponents_using_maxima_IQR(noisedimage, Colocalization.tomask(ccs))
	out = maskoutline(Colocalization.tomask(cmsk));
	roiI, roiA, roiS = [slicerois(i, roibox) for i in [noisedimage, A, out]]
	# RI=1
	for RI in 1:length(roiS)
		_i = SPECHT.tcolors( [ roiI[RI]./maximum(roiI[RI]), roiA[RI]./maximum(roiA[RI]), roiS[RI]./maximum(roiS[RI])])
		Images.save(joinpath(output, "annotate_roi_$(RI)_noise_$j.tif"), _i)

		_i = roiI[RI]
		Images.save(joinpath(output, "annotate_roi_$(RI)_noise_$(j)_gt.tif"), _i)
	end
	results[NSFAC] = ccs, out, noisedimage, cmsk, roiI, roiA, roiS
end


ccs, out, ni, cmsks, roiI, roiA, roiS=results[facs[2]]

imshow(SPECHT.tcolors([ni, Colocalization.tomask(A), SPECHT.maskoutline(Colocalization.tomask(ccs))]))
imshow(SPECHT.tcolors([ni, Colocalization.tomask(A), out]))
