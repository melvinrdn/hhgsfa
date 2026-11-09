# StrongFieldApproximation.jl side of the benchmark: one induced dipole per line of cases.txt
# (id I0[W/cm²] fwhm[fs] ndt window), saved with its field to out_jl/<id>.npz. Argon, 1030 nm, truncated Gaussian.
using StrongFieldApproximation, ElectricFields, Unitful, UnitfulAtomic, NPZ

const IP = 0.5792  # argon [Ha], as hhgsfa.sfa.IP_AU

function pulse(I0, fwhm)
    @field(F) do
        λ = 1030.0u"nm"
        I₀ = I0 * u"W/cm^2"
        τ = fwhm * u"fs"
        toff = 1.5fwhm * u"fs"
        tmax = 2fwhm * u"fs"
        env = :trunc_gauss
    end
    F
end

memory(ndt, window) = window == "first" ? ndt : typemax(Int)  # first return: excursions up to one cycle

function run(dir)
    out = joinpath(dir, "out_jl")
    mkpath(out)
    compile = @elapsed induced_dipole(IP, pulse(1e14, 5.0), 32; memory=32, verbosity=0)
    npzwrite(joinpath(out, "compile.npz"), Dict("time" => compile))
    for line in eachline(joinpath(dir, "cases.txt"))
        id, I0, fwhm, ndt, window = split(line)
        file = joinpath(out, id * ".npz")
        isfile(file) && continue
        I0, fwhm, ndt = parse(Float64, I0), parse(Float64, fwhm), parse(Int, ndt)
        F = pulse(I0, fwhm)
        t = collect(timeaxis(F, ndt))
        time = @elapsed d = induced_dipole(IP, F, ndt; memory=memory(ndt, window), verbosity=0)
        npzwrite(file, Dict("t" => t, "F" => field_amplitude.(F, t), "A" => vector_potential.(F, t), "d" => d, "time" => time))
        println(id, ": ", length(t), " points, ", round(time, digits=2), " s")
    end
end

run(@__DIR__)
