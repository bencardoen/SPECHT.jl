
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
using Images, Colors, DataFrames, CSV, Statistics, LinearAlgebra
import Glob
import Colocalization
using Logging
import ImageMagick
import ImageFiltering
import Random
using ProgressMeter
using Distributions
using Dates
using ArgParse
using LoggingExtras
using Dates
using Base.Threads

function tryparseseries(strng)
	# Serie_xyz
	@assert occursin("Serie", strng)
	parts = split(strng, '_')
	@assert(length(parts) == 2)
	return tryparse(Int64, parts[2])
end


function process_dir(root, selector, channels, outdir, oldbehavior, z, SQR, selfscale, mindistance, maxdistance; mode="nosegment", sigma=3, edgeiters=20, PRC=3.75, quantile=.9)
    @assert(isdir(outdir))
    contents = readdir(root)
	@info "Reading $root"
    result = DataFrames.DataFrame(celln=String[], serie=String[], experiment=String[],
	c1=Int64[], c2=Int64[], c12=Int64[], c12mito=Int64[], c1mito = Int64[], c2mito=Int64[],
	cellarea=Float64[], anisotropy=Float64[], major=Float64[], minor=Float64[])
    spots = DataFrames.DataFrame(celln=String[], serie=String[], experiment=String[],channel=String[],area=Int64[],
	distancec1c2mito=Float64[], adjacent_mito_spot_count=Int64[], mean_intensity_adjacent_mito=Float64[],
	std_intensity_adjacent_mito=Float64[], mindistancenearestC12=Float64[], mean_intensity_spot=Float64[], std_intensity_spot=Float64[], intensity_effect_size=Float64[],
	mean_intensity_channel=Float64[], std_intensity_channel=Float64[], mean_intensity_nearest_mito=Float64[], mean_intensity_mitochannel=Float64[],
	size_nearest_mito=Int64[], mito_pixels_under_c12spot=Float64[], mean_int_mito_under_c12=Float64[], std_int_mito_under_c12=Float64[])
	emptyspots = copy(spots)
	emptyresults = copy(result)
	### Add mito channel
    clabel = ["C1", "C2", "C12", "C12M", "C1M", "C2M", "Mito"]
	ecount = 0
	totalcells = 0
    for experiment in contents
        if occursin(selector, experiment)
            @info "Processing --> $selector"  # experiment
            for subdir in readdir(joinpath(root, experiment))
				if occursin("Cropped", subdir)
                    series = readdir(joinpath(root, experiment, subdir))
					@info "Processing total of $(length(series)) Series"
					for seriedir in series
						serie = tryparseseries(seriedir)
						if serie == -1
							@error "Skipping, could not read decode serie from $(seriedir)"
							continue
						end
						cells = readdir(joinpath(root, experiment, subdir, seriedir))
						ncells = length(cells)
						@info "Processing total of $ncells cells"
						totalcells += ncells
						resultarray = [copy(emptyresults) for i in 1:ncells]
						spotarray = [copy(emptyspots) for i in 1:ncells]
						p = Progress(ncells, .5)
						@threads for cellindex in 1:ncells #enumerate crashes when threaded
							celln = cells[cellindex]
	                        @debug "Found cell # $celln in $subdir and experiment $experiment"
	                        qpath = joinpath(root, experiment, subdir, seriedir, celln)
							res, _ecount, _error, images = process_cell(qpath, channels, outdir, serie, subdir, experiment, z, selfscale, celln, SQR,
								mode=mode, sigma=sigma, edgeiters=edgeiters, PRC=PRC, quantile=quantile)
							ecount += _ecount
							if _ecount > 0
								@debug "Invalid data, skipping postprocessing Cell $celln Serie $serie Exp $experiment"
								_failresult = copy(emptyresults)
								push!(_failresult, [celln, string(serie), experiment, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
								resultarray[cellindex] = _failresult
								continue
							end
							res[3] = images
	                        c1count, c2count, c1c2count, count_overlap_mito, count_overlap_c1_mito, count_overlap_c2_mito, areas, distances, union_mask_c12, intensities, channelstats = qspot(res, oldbehavior)
							@assert length(areas) == length(clabel)
							@assert length(intensities) == 3 == length(channelstats)
							tricolor_result_no_union, tricolor_result_union = computemasks(res, union_mask_c12)
							mindistances, nearestc12 = quantify_adjacent_c12(res[0][2], union_mask_c12)
							c12_m_adjacencies, c12_m_adjacent_mask = quantify_adjacent_mitophagy(union_mask_c12, res[0][end], images[0], mindistance, maxdistance)
							QNM, QNM_image = quantify_nearest_mitophagy(union_mask_c12, res[0][end], images[0])
							Images.save(joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_c12_nearest_mito_pairs_filtered_$(SQR).tif"), Gray{N0f16}.(QNM_image))
							Images.save(joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_c12_nearest_mito_filtered_$(SQR).tif"), nearestc12)
							Images.save(joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_red_channel_1_green_channel_2_blue_channel_0_non_union_mask_filtered_$(SQR).tif"), RGB{N0f16}.(tricolor_result_no_union))
							Images.save(joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_C12_adjacent_mitos_$(SQR).tif"), Gray{N0f16}.(c12_m_adjacent_mask))
							Images.save(joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_red_channel_1_green_channel_2_blue_channel_0__c12union_mask_filtered_$(SQR).tif"), RGB{N0f16}.(tricolor_result_union))

							rawimages = res[5]
							out3 = tricoloroutline([rawimages[i] for i in [1,2,0]], [res[i][end] for i in [1,2,0]])
							Images.save(joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_red_channel_1_green_channel_2_blue_channel_0_outline_on_raw_images_$(SQR).tif"), out3)
							ym_rm, yomo_rm = visout(union_mask_c12, res[0][end], rawimages[0])
							Images.save(joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_red_channel_1_green_channel_2_blue_channel_0_yellow_mask_on_raw_mito_$(SQR).tif"), ym_rm)
							Images.save(joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_red_channel_1_green_channel_2_blue_channel_0_yellow_blue_outlines_on_raw_mito_$(SQR).tif"), yomo_rm)
							cellwisestats = [c1count, c2count, c1c2count, count_overlap_mito, count_overlap_c1_mito, count_overlap_c2_mito]
							metadata = Dict("experiment" => experiment, "celln" => celln, "serie" => serie)
							mito_under_c12m = quantify_c12_mito_overlap(union_mask_c12, res[0][end], rawimages[0])
							## res[0-2][0] fused
							fused = [Float64.(images[i]) for i in [1,2,0]]
							cellimage = reduce(.+, fused)
							cellimage = cellimage ./ maximum(cellimage)
							### Compute shape / area stats here

							cellstatsdict = cellstats(cellimage)
							metadata = merge(cellstatsdict, metadata)
							Images.save(joinpath(outdir, "$(experiment)_serie_$(serie)_celln_$(celln)_fused_image_$(SQR).tif"), Images.N0f16.(cellimage))
							_spots, _results = record_stats(areas; clabel=clabel, metadata=metadata, distances=distances, channelstats=channelstats, c12_m_adjacencies=c12_m_adjacencies, mindistances=mindistances, intensities=intensities, QNM=QNM, cellwisestats=cellwisestats,
							emptyspots=emptyspots, emptyresults=emptyresults, mito_under_c12m=mito_under_c12m, cellimage=cellimage, zeromito=iszero(images[0]))

							spotarray[cellindex] = _spots
							resultarray[cellindex] = _results
							next!(p)
						end ### End of cells
						result = vcat([result, vcat(resultarray...)]...)
						spots = vcat([spots, vcat(spotarray...)]...)
                    end ### End of serie
                end
            end
        else
			@debug "No such $selector in $experiment"
		end
    end
	ecount > 0 ? (@error "ERROR Total errors : $ecount / $(totalcells*3) images out of $totalcells cells with processed") : nothing
    return result, spots
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--inpath", "-i"
            help = "input folder"
            arg_type = String
            required = true
		"--outpath", "-o"
            help = "output folder"
            arg_type = String
            required = true
        "--prc", "-p"
            help = "Precision recall balance. When filtermode == autotune, value > 1 will favor recall, < 1 precision, default 1"
            arg_type = Float64
            required = false
            default = 3.75
		"--edge-width"
		    help = "Relative width of cell edge to be removed. Only relevant if segment is true. Default 20 (~10-20 px)"
            arg_type = Int64
            required = false
            default = 20

		"--min-distance-mito"
			help = "Minimum distance between C1C2 spots <-> mito to be considered in contact (pixels, default = 0.0)"
            arg_type = Float64
            required = false
            default = 0.0
		"--max-distance-mito"
			help = "Maximum distance between C1C2 spots <-> mito , between [min,max] of each mito spot in contact will be quantified (intensity), default 1.0"
            arg_type = Float64
            required = false
            default = 1.0
        "--filterleq", "-f"
            help = "Filter objects < fxf pixels, default 5"
            arg_type = Int64
            required = false
            default = 5
        "--zmin", "-z"
            help = "min z score to segment (μ + z σ)"
            arg_type = Float64
            default = 1.75
		"--sigma"
			help = "σ for LoG smoothing, defaults to 3.0 (use float notation)"
            arg_type = Float64
            default = 3.0
		"--skip-mitochondria"
            help = "If set, skip the mitochondria channel (and thus all its related stats -> 0), default false"
            action = :store_true
		"--mode"
            help = "If input data is not segmented and has significant background, set to 'segment' (deprecated) or 'apery'. Cell segmentation only works if 1 cell is FOV with partials, not multiple cells."
            arg_type = String
			default = "default"
		"--denoise"
            help = "[0.25-0.9] Controls how sensitive cell segmentation is in apery mode."
            arg_type = Float64
			default = 0.9
		"--max-spot-size"
            help = "Set an upper limit k, s.t. spots of size >  kxk are dropped, default +∞"
            arg_type = Float64
			default = Inf64
		"--zmax", "-Z"
			help = "max z score to segment (μ + z σ)"
			arg_type = Float64
			default = 1.75
		"--zstep", "-s"
			help = "step interval from zmin to zmax, if only 1 z value is to be tested, set -z x -Z x -s anything > 0"
			arg_type = Float64
			default = 0.25

    end

    return parse_args(s)
end

function run()
    date_format = "yyyy-mm-dd HH:MM:SS"

    timestamp_logger(logger) = TransformerLogger(logger) do log
      merge(log, (; message = "$(Dates.format(now(), date_format)) $(basename(log.file)):$(log.line): $(log.message)"))
    end
    ConsoleLogger(stdout, Logging.Info) |> timestamp_logger |> global_logger
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        @info "  $arg  =>  $val"
    end
	cmin = 0 ## Changing this to a variable is not worth the effort (for now)
	channels = [cmin, cmin+1, cmin+2]
    dataroot = parsed_args["inpath"]
	@assert(isdir(dataroot))
	_outbase = parsed_args["outpath"]
	minmito = parsed_args["min-distance-mito"]
	maxmito = parsed_args["max-distance-mito"]
	@assert 0 <= minmito <= maxmito
    zmin = parsed_args["zmin"]
	zmax = parsed_args["zmax"]
	zstep = parsed_args["zstep"]
	@assert 0 < zmin <= zmax
	@assert zstep > 0
	zs = zmin:zstep:zmax |> collect
	@assert(length(zs) >= 1)
	SQR = parsed_args["filterleq"]
	@assert SQR >= 0
	@info "Using dataroot $(dataroot)"
	subdirs = readdir(dataroot)
	@info "Have $(length(subdirs)) subdirectories of $dataroot"
	mkpath(_outbase)
	@info "Output root = $(_outbase)"
	selfscale = false
	if ! isinf(parsed_args["max-spot-size"])
		@error "Not supported yet"
		error("Max spot size not supported yet")
	end
	for (exi, replicate) in enumerate(subdirs)
		@info "Replicate nr $replicate"
		repnr = tryparse(Int, replicate)
		outpath = joinpath(_outbase, "replicate_$(replicate)")
	 	indir = joinpath(dataroot, replicate)
		choices = readdir(indir)
		for z in zs
			@debug "Using Z = $z"
			outpath_z = "$(outpath)_$(z)"
		    checkdir(outpath_z)
			@info "Using outpath $(outpath_z)"
		    for _s in choices
		        @info "Processing celltype $(_s) with z $(z) in replicate $(replicate)"
		        @assert(isdir(indir))
		        dfxcounts, dfxspots = process_dir(indir, _s, channels, outpath_z, false, z, SQR, selfscale, minmito, maxmito, mode=parsed_args["mode"], sigma=parsed_args["sigma"], edgeiters=parsed_args["edge-width"], PRC=parsed_args["prc"], quantile=parsed_args["denoise"])
		     	insertcols!(dfxcounts, size(dfxcounts, 2), (:z => z))
		        insertcols!(dfxcounts, size(dfxcounts, 2), (:replicate => repnr))
		        insertcols!(dfxspots, size(dfxspots, 2), (:z => z))
		        insertcols!(dfxspots, size(dfxspots, 2), (:replicate => repnr))
		        CSV.write(joinpath(_outbase, "$(_s)counts_$(z)_replicate_$(replicate).csv"), dfxcounts)
		        CSV.write(joinpath(_outbase, "$(_s)spots_$(z)_replicate_$(replicate).csv"), dfxspots)
			end
		end
	end
	@info "Finished"
end


run()
