# using Pkg;
# Pkg.activate(".")
using Images, SPECHT, Glob
using Colocalization
using Logging
using ImageView
using Images, Colors, DataFrames, CSV, Statistics, LinearAlgebra
import Glob
using Logging
using Dates
using ArgParse
using LoggingExtras
using Dates
using Base.Threads

function autophagy(dir, z=1.75, sigma=3, pattern="*[1,2].tif", SQR=5)
    tifs = Glob.glob(pattern, dir)
    @info "$tifs"
    images = [Images.load(i) for i in tifs]
    if length(images)!=2
        @error "No matching files in $dir"
        return nothing
    end
    IMG = images[1] .+ images[2]
	CM = Colocalization.tomask(IMG)
	msk = Colocalization.tomask(iterative(CM, ImageMorphology.dilate, SQR))
	edge = ⨣(𝛁(msk))
	edged = Colocalization.tomask(iterative(edge, ImageMorphology.dilate, SQR*2))
    results = Dict()
    for (i, image) in enumerate(images)
        ccs, imgl, Tg, _img, _msk = process_tiffimage(image, z, [sigma, sigma], false, 2, 0, edgemask=edged)
        cmsk = filter_cc_sqr_greater_than(ccs, _img, SQR)
        ccs = Images.label_components(cmsk, trues(3,3))
        results[i] = ccs, cmsk
    end
    return msk, edged, images, results
end

function quantify(results)
    @info "Quantifying stats"
    c1, m1 = results[1]
    c2, m2 = results[2]
    N1=maximum(c1)
    N2=maximum(c2)
    d12= pairwise_distance(c1, m2)
    d21= pairwise_distance(c2, m1)
    a1 = Images.component_lengths(c1)[2:end]
    a2 = Images.component_lengths(c2)[2:end]
    df1 = DataFrame(distance_to_other=d12, area=a1, channel=1)
    df2 = DataFrame(distance_to_other=d21, area=a2, channel=2)
    return vcat(df1, df2)
end

function pairwise_distance(from_cc, to_mask)
    dismap = Images.distance_transform(Images.feature_transform(Bool.(to_mask)))
    N = maximum(from_cc)
    @info "Have $N components"
    dis = zeros(maximum(from_cc))
    dis .= Inf
    ind = Images.component_indices(from_cc)[2:end]
    for i in 1:N
        @inbounds dis[i] = minimum(dismap[ind[i]])
    end
    return dis
end

function process_dir(indir, outdir, z=1.75, sigma=3, pattern="*[1,2].tif", SQR=5)
    dfs = []
    for replicate in readdir(indir)
        for treatment in readdir(joinpath(indir, replicate))
            for cellnumber in readdir(joinpath(indir, replicate, treatment))
                @info "Replicate $replicate Treatment $treatment Cell $cellnumber"
                celldir=joinpath(indir, replicate, treatment, cellnumber)
                cell, edge, imgs, results = autophagy(celldir, z, sigma, pattern, SQR);
                Images.save(joinpath(outdir, "Replicate_$(replicate)_Treatment_$(treatment)_Cell_$(cellnumber)_cellmask.tif"), cell)
                for k in keys(results)
                    spots_k = results[k][2]
                    Images.save(joinpath(outdir, "Replicate_$(replicate)_Treatment_$(treatment)_Cell_$(cellnumber)_spots_channel_$(k).tif"), spots_k)
                end
                dfx= quantify(results)
                dfx[!, "replicate"] .= replicate
                dfx[!, "cellnumber"] .= cellnumber
                dfx[!, "treatment"] .= treatment
                push!(dfs, dfx)
            end
        end
    end    
    DFX = vcat(dfs)
    CSV.write(joinpath(outdir, "table_spots.csv"), DFX)
end


# process_dir(topdir, outdir)

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
        "--filterleq", "-f"
            help = "Filter objects < fxf pixels, default 5"
            arg_type = Int64
            required = false
            default = 5
        "--zval", "-z"
            help = "min z score to segment (μ + z σ)"
            arg_type = Float64
            default = 1.75
		"--sigma"
			help = "σ for LoG smoothing, defaults to 3.0 (use float notation)"
            arg_type = Float64
            default = 3.0
        "--pattern"
			help = "Pattern of channels, default *[1,2].tif to pick up channels 1 and 2"
            arg_type = String
            default = "*[1,2].tif"
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
    println("Arugments are:")
    for (arg,val) in parsed_args
        @info "  $arg  =>  $val"
    end
    process_dir(parsed_args["inpath"], parsed_args["outpath"], parsed_args["zval"], parsed_args["sigma"], parsed_args["pattern"], parsed_args["filterleq"])
end

run()