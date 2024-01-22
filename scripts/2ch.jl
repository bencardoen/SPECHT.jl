# using Pkg;
# Pkg.activate(".")
using Images, SPECHT, Glob
using Colocalization
using Logging
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
# p = "/home/bcardoen/storage/parsa/P/test2channel/1/HT CCCP -BaF siAMF/001"
# msk, ed, ims, results = autophagy(p)

# i1, i2 = ims
# m1, m2 = results[1][2], results[2][2]
# object_stats(i1, i2, m1, m2)



#     c1stats = Colocalization.describe_cc(c1, i1)
#     c2stats = Colocalization.describe_cc(c2, i2)
#     # df1 = DataFrame(channel=1, overlap=o12, distance_to_nearest=disc1_c2, area=Images.component_lengths(c1)[2:end], mean=c1stats[:,1], std=c1stats[:,2])
#     # df2 = DataFrame(channel=2, overlap=o21, distance_to_nearest=disc2_c1, area=Images.component_lengths(c2)[2:end], mean=c2stats[:,1], std=c2stats[:,2])
#     # return vcat(df1, df2), d1map, d2map
# # end

# c1 = results[1][1]
# m1 = results[1][2]
# c2 = results[2][1]
# m2 = results[2][1]

# dfx = quantify(results, ims)
# @info dfx
# # vcat([dfx]...)

function overlap(cfrom, mto)
    ov = zeros(maximum(cfrom))
    for (i,ind) in enumerate(Images.component_indices(cfrom)[2:end])
        ov[i] = sum(mto[ind])
    end
    return ov
end

function quantify(results, images)
    @info "Quantifying stats"
    c1, m1 = results[1]
    c2, m2 = results[2]
    @info "Channel 1 to channel 2"
    d12= pairwise_distance(c1, m2)
    @info "Channel 2 to channel 1"
    d21= pairwise_distance(c2, m1)
    a1 = Images.component_lengths(c1)[2:end]
    a2 = Images.component_lengths(c2)[2:end]
    c1stats = Colocalization.describe_cc(c1, images[1])
    c2stats = Colocalization.describe_cc(c2, images[2])
    ov12 = overlap(c1, m2)
    ov21 = overlap(c2, m1)
    df1 = DataFrame(distance_to_other=d12, area=a1, mean_intensity=c1stats[:,1], std_intensity=c1stats[:,2], overlap_other=ov12)
    df1[!, "channel"] .= 1
    df2 = DataFrame(distance_to_other=d21, area=a2, mean_intensity=c2stats[:,1], std_intensity=c2stats[:,2], overlap_other=ov21)
    df2[!, "channel"] .= 2
    return vcat(df1, df2)
end

# function group_data(df, overlap_minimum=0, minimum_size=0)
#     @info "Filtering with minimum overlap $(overlap_minimum) and minimum size $(minimum_size) total number of rows x cols $(size(df))"
#     _df = copy(df)
#     _df = filter(row -> row.area >= minimum_size, _df)
#     grouped_df = groupby(_df, [:cellnumber, :treatment, :channel, :replicate])
#     gdf = combine(grouped_df, nrow .=> :nr_spots)
#     c1c2df = filter(row -> row.channel == 1 && row.overlap_other > overlap_minimum, _df)
#     c12df = combine(groupby(c1c2df, [:cellnumber, :treatment, :channel, :replicate]), nrow .=> :nr_spots)
#     c12df.channel .= 12
#     return vcat(c12df, gdf)
# end

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

function process_dir(indir, outdir, z=1.75, sigma=3, pattern="*[1,2].tif", SQR=5, minoverlap=0)
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
                dfx= quantify(results, imgs)
                dfx[!, "replicate"] .= replicate
                dfx[!, "cellnumber"] .= cellnumber
                dfx[!, "treatment"] .= treatment
                push!(dfs, dfx)
            end
        end
    end    
    DFX = vcat(dfs...)
    @info "Saving tabular results in $(joinpath(outdir, "table_spots.csv"))"
    CSV.write(joinpath(outdir, "table_spots.csv"), DFX)
    @info "Summarizing"
    DFS = group_data(DFX, minoverlap, SQR)
    CSV.write(joinpath(outdir, "summarized.csv"), DFS)
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
        "--min_overlap"
			help = "Minimum overlap of objects to be considered to be colocalizing"
            arg_type = Int64
            default = 0
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
    println("Arguments are:")
    for (arg,val) in parsed_args
        @info "  $arg  =>  $val"
    end
    process_dir(parsed_args["inpath"], parsed_args["outpath"], parsed_args["zval"], parsed_args["sigma"], parsed_args["pattern"], parsed_args["filterleq"], parsed_args["min_overlap"])
end

run()