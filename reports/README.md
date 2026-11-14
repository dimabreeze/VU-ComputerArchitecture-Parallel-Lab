# Etapų ataskaitos

Darbo eiga žingsnis po žingsnio. Kiekvieno etapo grafikai yra `<stage>/plots/`, o
`README.md` aprašo, ką pamatėme, kodėl ir prie ko tai privedė. Grafikai iš naujo generuojami iš
`results/csv/` komanda `make reports` (visi etapai) arba `make reports STAGES="02 05"`;
etapų apibrėžimai (duomenys, filtrai, pavadinimai) yra
[`scripts/reports.sh`](../scripts/reports.sh).

| etapas | tema |
|-------|-------|
| [01-baseline](01-baseline/README.md) | visų keturių technologijų bazinės realizacijos |
| [02-tlb-cliff](02-tlb-cliff/README.md) | 3× sulėtėjimas viršijus n ≈ 3500 ir jo priežastis |
| [03-seq-opt](03-seq-opt/README.md) | nuoseklaus kodo optimizavimas: 125× viename branduolyje |
| [04-openmp](04-openmp/README.md) | OpenMP: base → opt → opt2 |
| [05-mpi](05-mpi/README.md) | MPI: base → opt |
| [06-opencl](06-opencl/README.md) | OpenCL: naive → coalesced → tiled → regblock |
| [07-amdahl](07-amdahl/README.md) | nuoseklioji dalis ir Amdalo dėsnis: teorija ir matavimai, E branduoliai |
| [08-final](08-final/README.md) | galutinis geriausių variantų palyginimas |

Pristatymai (lietuvių k.): [`pptx/`](pptx/).

Kompiuteris: Apple M2 Pro, 8 našieji (P) + 4 taupieji (E) branduoliai, 19 branduolių GPU, 16 KB puslapiai,
bendroji CPU ir GPU atmintis. Laikai – pakartotinių paleidimų medianos.
