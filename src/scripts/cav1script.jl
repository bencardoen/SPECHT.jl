# This file contains function used for the Cav1 analysis for the SPECHT paper.

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
using DataFrames

## Parameter sweep
const PREC_REC = 2
const SIGMA = 1.5

### Function to compare PTRF mean SP correlation with Cav1 mask
function spotcorrelate(ccs, img1, img2)
    """
    Compute spot correlation of connected components over 2 channels.
    Returns a mask with the spearman correlation in each spot, a vector Nx2 of correlation, mean img2/spot and a mask of that mean/spot.
    """
    NL = maximum(ccs)
    indices = Images.component_indices(ccs)[2:end]
    res = zeros(NL, 2)
    mask = Float64.(aszero(img1))
    mask2 = Float64.(aszero(img1))
    for component in 1:NL
        @inbounds ind = indices[component]
        @inbounds view1 = img1[ind]
        @inbounds view2 = img2[ind]
        srt1 = sortperm((view1[:]))
        srt2 = sortperm((view2[:]))
        spc = cor(srt1, srt2)
        @inbounds mask[ind] .= spc
        mv2 = Statistics.mean(view2)
        @inbounds mask2[ind] .= mv2
        @inbounds res[component, 1] = spc
        @inbounds res[component, 2] = mv2
    end
    return mask, res, mask2
end

### Processing function, kept for repro, superceded by SPECHT.process_tiffimage
function process_cav_file(indir, regex)
    recall_precision_balance = PREC_REC
    sigmas = [0,SIGMA] # If decon, first sigma = 0
    z = 2
    smooth = 0
    tiff = Glob.glob(regex, indir)
    println("Found $(length(tiff)) tiffs using regex $(regex)")
    if length(tiff) != 1
        println("[EE] -- Ambiguous contents, skipping")
    else
        println("Processing $(tiff[1])")
        cccs, cimgl, cTg, cimg, _ = process_tiff(tiff[1], z, sigmas, true, recall_precision_balance, smooth)
        return cccs, cimgl, cTg, cimg
    end
end

# For the labelling section, please see the updated API in the README.md
