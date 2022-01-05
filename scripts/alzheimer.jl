# This file contains function used for the Alzheimer dataset analysis for the SPECHT paper.
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
using SPECHT
using Images, Colors, ImageView, DataFrames, CSV, Statistics, LinearAlgebra
import Glob
import ImageMagick
import ImageFiltering
import Random
using Colors
using Gtk.ShortNames
using Distributions

###
# Parsing script 2020 06 02 -- git version 47a0c3b605128631169b03538ff16f27e80b7e5f
# Reads in Eye data, builds 2 models to contrast AD+ data, then builds a joint model
###


#### NOTE : see the README.md for easier to understand examples on how to use the API.
#### NOTE 2: The below code is meant to be executed as Hydrogen (Atom, probably VC Code) blocks (similar to Jupyter notebook), for maximum control.

function tocolor(img, color)
    if color == 1
        return colorview(RGB, img, aszero(img), aszero(img))
    end
    if color == 2
        return colorview(RGB, aszero(img), img, aszero(img))
    end
    if color == 3
        return colorview(RGB, aszero(img), aszero(img), img)
    end
end


root = "SET_INDIR_HERE"
@assert(isdir(root))

NX = 2 # change to 2 to build model on t2

OUTROOT = "SET_OUTPUT_DIR_HERE"
outpath = joinpath(OUTROOT, "trained_$(NX)")
mkpath(outpath)
@assert(isdir(outpath))




### 0. Load the tiff files
tiffs_i1_norm = Glob.glob("*.tif", joinpath(root, "Imager1", "Normal"))
tiffs_i2_norm = Glob.glob("*.tif", joinpath(root, "Imager2", "Normal"))
tiffs_i2_ds = Glob.glob("*.tif", joinpath(root, "Imager2", "AD"))

i1_n_tiffs = [Images.load(t) for t in tiffs_i1_norm]
i2_n_tiffs = [Images.load(t) for t in tiffs_i2_norm]
i2_d_tiffs = [Images.load(t) for t in tiffs_i2_ds]

## Save them to temp storage
_tempdir = mktempdir()
for (it,ti) in enumerate(i1_n_tiffs)
    # Images.save(joinpath(tempdir, "$(it).tif"),channelview(ti)[1,:,:])
    Images.save(joinpath(_tempdir, "$(it)_n1.tif"),ti)
end
for (it,ti) in enumerate(i2_d_tiffs)
    # Images.save(joinpath(tempdir, "$(it)_d.tif"),channelview(ti)[1,:,:])
    Images.save(joinpath(_tempdir, "$(it)_d2.tif"),ti)
end
for (it,ti) in enumerate(i2_n_tiffs)
#     Images.save(joinpath(_tempdir, "$(it)_n2.tif"),ti)
    Images.save(joinpath(_tempdir, "$(it)_n2.tif"),ti)
end

## Configure SPECHT
recall_precision_balance = 1.5 #1.5
sigmas = [1,1]
z = 2 # Not in use
res1 = Dict()
res2 = Dict()

# Build the objects for Normal 1
for (it,tf) in enumerate(i1_n_tiffs)
    println("$(it)_n1.tif")
    cccs, cimgl, cTg, cimg = process_tiff(joinpath(_tempdir, "$(it)_n1.tif"), z, sigmas, true, recall_precision_balance, 0)
    println("Summarizing")
    _rmc = summarize_spots(cimg, rlap(cimgl), cccs)
    res1[it] = copy(cccs), copy(cimgl), copy(cTg), copy(cimg), copy(_rmc)
end


## Build the objects for Normal 2
for (it,tf) in enumerate(i2_n_tiffs)
    println("$(it)_n1.tif")
    cccs, cimgl, cTg, cimg = process_tiff(joinpath(_tempdir, "$(it)_n2.tif"), z, sigmas, true, recall_precision_balance, 0)
    println("Summarizing")
    _rmc = summarize_spots(cimg, rlap(cimgl), cccs)
    res2[it] = copy(cccs), copy(cimgl), copy(cTg), copy(cimg), copy(_rmc)
end


# Build objects for AD+
resd = Dict()
for (it,tf) in enumerate(i2_d_tiffs)
    cccs, cimgl, cTg, cimg = process_tiff(joinpath(_tempdir, "$(it)_d2.tif"), z, sigmas, true, recall_precision_balance, 0)
    println("Summarizing")
    _rmc = summarize_spots(cimg, rlap(cimgl), cccs)
    resd[it] = copy(cccs), copy(cimgl), copy(cTg), copy(cimg), copy(_rmc)
end


if NX == 1
    println("NX = 1, using 1 as training model")
    res = copy(res1)
else
    println("NX = 2, using 2 as training model")
    res = copy(res2)
end



## Distance transform of features to model
sp_x = vcat([res[k][end] for k in keys(res)]...)
zs_control, ms_control = computemahabdistances(sp_x)
mean_control, std_control = mean(zs_control), std(zs_control)

## Normal Control : Self - similarity

# controlpxs = cantelli(zs_control, mean_control, std_control)
# Probability of spot in control being from control
res_cnt = Dict()
dfs_cnt = []
for (i,k) in enumerate(keys(res))
    da_mb = res[k][end]
    ccs = res[k][1]
    img = res[k][4]
    _rmc = res[k][end]
    z_control_to_control = computemahabdistances_vector(da_mb, sp_x)[1]
    p_x_i = cantelli(z_control_to_control, mean_control, std_control)
    mki = annotate_spots(img, ccs, p_x_i)
    res_cnt[k] = z_control_to_control, p_x_i, mki, img
    push!(dfs_cnt, DataFrames.DataFrame(cellnr=k, z_to_control=z_control_to_control, p_alz=p_x_i, area=_rmc[:,1],intensity=_rmc[:,2], laplacian=_rmc[:,3]))
end
DF_CNT = vcat(dfs_cnt...)

CSV.write(joinpath(outpath, "control_cells_imager_$(NX).csv"), DF_CNT)


# Visualize and SAVE
cvcs = [res_cnt[k][4] for k in keys(res_cnt)]
cmks = [abs.(0 .- res_cnt[k][end-1]) for k in keys(res_cnt)]
ccmks = [tocolor(k, false) for k in cmks]
for i in 1:length(cmks)
    Images.save(joinpath(outpath, "raw_$(i)_normal_imager_$(NX).tif"), cvcs[i])
    ctm = colorview( RGB, flip(cmks[i]), cmks[i], aszero(cvcs[i]))
    Images.save(joinpath(outpath, "normal_$(NX)_labelled_red_abnormal_green_normal_$(i)_using_$(NX).tif"), ctm)
end


res_other_result = Dict()
dfs_other = []
if NX == 1
    print("Switching to res2")
    res_other = copy(res2)
else
    print("Switching to res1")
    res_other = copy(res1)
end
@assert(res != res_other)
for (i,k) in enumerate(keys(res_other))
    da_mb = res_other[k][end]
    ccs = res_other[k][1]
    img = res_other[k][4]
    _rmc = res_other[k][end]
    z_other_to_control = computemahabdistances_vector(da_mb, sp_x)[1]
    p_x_i = cantelli(z_other_to_control, mean_control, std_control)
    mki = annotate_spots(img, ccs, p_x_i)
    res_other_result[k] = z_other_to_control, p_x_i, mki, img
    push!(dfs_other, DataFrames.DataFrame(cellnr=k, z_to_control=z_other_to_control, p_alz=p_x_i, area=_rmc[:,1],intensity=_rmc[:,2], laplacian=_rmc[:,3]))
end
DF_OTHER = vcat(dfs_other...)
if NX == 1
    CMX = 2
else
    CMX = 1
end
CSV.write(joinpath(outpath, "normal_cells_imager_$(CMX).csv"), DF_OTHER)


ovcs = [res_other_result[k][4] for k in keys(res_other_result)]
omks = [abs.(0 .- res_other_result[k][end-1]) for k in keys(res_other_result)]
ocmks = [tocolor(k, false) for k in omks]

for i in 1:length(ocmks)
    Images.save(joinpath(outpath, "raw_$(i)_normal_images_$(CMX).tif"), ovcs[i])
    ctm = colorview( RGB, flip(omks[i]), omks[i], aszero(ovcs[i]))
    Images.save(joinpath(outpath, "normal_$(CMX)_labelled_red_abnormal_green_normal_$(i)_using_$(NX).tif"), ctm)
end


### Alzheimer

# P in alzheimer
res_alz = Dict()
dfs = []
for (i,k) in enumerate(keys(resd))
    da_mb = resd[k][end]
    ccs = resd[k][1]
    img = resd[k][4]
    _rmc = resd[k][end]
    z_alz_to_control = computemahabdistances_vector(da_mb, sp_x)[1]
    p_x_i = cantelli(z_alz_to_control, mean_control, std_control)
    mki = annotate_spots(img, ccs, p_x_i)
    res_alz[k] = z_alz_to_control, p_x_i, mki, img
    push!(dfs, DataFrames.DataFrame(cellnr=k, z_to_control=z_alz_to_control, p_alz=p_x_i, area=_rmc[:,1],intensity=_rmc[:,2], laplacian=_rmc[:,3]))
end
DF_ALZ =  vcat(dfs...)
CSV.write(joinpath(outpath, "alzheimer_cells_trained_using_$(NX).csv"), DF_ALZ)

# DF_OTHER == DF_CNT
# Visualize Alzheimer
avcs = [res_alz[k][4] for k in keys(res_alz)]
amks = [abs.(0 .- res_alz[k][end-1]) for k in keys(res_alz)]
acmks = [tocolor(k, false) for k in amks]
for i in 1:length(amks)
    Images.save(joinpath(outpath, "raw_$(i).tif"), avcs[i])
    ctm = colorview( RGB, flip(amks[i]), amks[i], aszero(avcs[i]))
    Images.save(joinpath(outpath, "alz_labelled_red_abnormal_green_normal_$(i)_using_$(NX).tif"), ctm)
end

### Postprocessing --> Joint model, run after both NX=1 and NX=2 have completed, with resd variables defined (e.g. at end of NX2 run)
using CSV
CSVS = Glob.glob("alz*.csv", joinpath(outpath, "trained_2"))
df2 = CSV.File(CSVS[1]) |> DataFrame

CSVS2 = Glob.glob("alz*.csv", joinpath(outpath, "trained_1"))
df1 = CSV.File(CSVS2[1]) |> DataFrame

@assert(! all(df1 == df2))

function combine_mass(r, s)
    top = s.*r
    denom = 1. .- (((1. .- s).*r + (1. .-r) .*s))
    return top/denom
end

function combne_weight(r, s)
    denom = 1. .- (((1. .- s).*r + (1. .-r) .*s))
    return log.(1. ./denom)
end

outpath_combined = joinpath(outpath, "combined")

res_combined = Dict()
_DX = []
for (i,k) in enumerate(keys(resd))
    ccs = resd[k][1]
    img = resd[k][4]
    _N = maximum(ccs)
    df13 = df1[df1.cellnr .== k, :]
    df23 = df2[df2.cellnr .== k, :]
    A = 1 .- max.(df13.p_alz, eps()) # if it's 1, 0, if it's
    B = 1 .- max.(df23.p_alz, eps())
    combined = combine_mass.(A, B)
    weight = combne_weight.(A, B)
    DFCOMB = DataFrame(p_alz_1=df13.p_alz, p_alz_2=df23.p_alz, p_alz_c=combined, w=weight , cellnr=k)
    push!(_DX, DFCOMB)
    @assert(size(combined, 1) == _N)
    mki = annotate_spots(img, ccs, combined)
    res_combined[k] = nothing, combined, mki, img
    cmki = tocolor(mki, true)
    Images.save(joinpath(outpath_combined, "alz_labelled_red_abnormal_green_normal_$(k)_using_$(NX)_combined.tif"), cmki)
    Images.save(joinpath(outpath_combined, "raw_$(k)_using_$(NX)_combined.tif"), img)
end
Combined_dataframe = vcat(_DX...)

CSV.write(joinpath(outpath_combined, "combined_results.csv"), Combined_dataframe)
