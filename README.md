# Lygiagreti matricų daugyba (OpenMP / MPI / OpenCL)

Užduotis: [docs/task.md](docs/task.md).  
Kurso medžiaga: [docs/Reference/](docs/Reference/).  
Analizė:
* [docs/tlb_cliff.md](docs/tlb_cliff.md) (staigus sulėtėjimas ties TLB riba), 
* [docs/amdahl.md](docs/amdahl.md) (Amdalo dėsnis, E branduoliai),
* [docs/memory.md](docs/memory.md) (atminties delsos, TLB ir pralaidumo matavimai).

Kompiuteris ir programinė įranga: [docs/machine.md](docs/machine.md).  

**Ataskaita (PDF):** [reports/ataskaita/](reports/ataskaita/) – bendroji dalis, 1 dalis OpenMP, 2 dalis MPI, 3 dalis OpenCL, bendros išvados ir visa ataskaita viename faile.  
Etapų ataskaitos: [reports/README.md](reports/README.md); 
Pristatymai: [reports/pptx/](reports/pptx/).

## Reikalavimai (macOS)

```
brew install libomp open-mpi
```

OpenCL yra macOS dalis.  
Grafikams reikia matplotlib (`make plot` automatiškai sukuria `.venv`).

## Struktūra

```
include/common.hpp     argumentai, Matrix, generatorius, laiko matavimas, tikrinimas, ataskaita (bendra)
include/gemm.hpp       greitas blokinis daugybos skaičiavimo branduolys (seq-opt, mpi-opt)
src/common/common.cpp
src/seq/base.cpp       nuoseklus etaloninis variantas (pagreitėjimo atskaitos taškas)
src/seq/opt.cpp        nuoseklus: blokavimas + pakavimas + mikrobranduolys registruose (docs/seq_opt.md)
src/openmp/base.cpp    OpenMP, griežtai pagal užduotį
src/openmp/opt.cpp     OpenMP, optimizuotas (i-k-j tvarka, simd)
src/openmp/opt2.cpp    OpenMP, blokavimas podėlyje ir registruose, dinaminis paskirstymas (docs/openmp_opt2.md)
src/mpi/base.cpp       MPI, ištaisytas kurso šablonas (docs/Reference/mpi_sample.md): pagal užduotį (matrix1 pildoma lygiagrečiai, Bcast, Send/Recv) (docs/mpi_opt.md)
src/mpi/opt.cpp        MPI, greitas branduolys (gemm.hpp) + eilučių blokai + Gatherv, pasirinktinis Ibcast konvejeris (docs/mpi_opt.md)
src/opencl/base.cpp    OpenCL pagrindinė programa; kerneliai src/opencl/base/*.cl (naive.cl)
src/opencl/opt.cpp     OpenCL pagrindinė programa: be kopijavimo, su patikrinimais, eilučių juostomis; kerneliai src/opencl/opt/*.cl (docs/opencl_opt.md)
src/opencl/common.cl   generatorius + pildymo kernelis, pridedamas prie kiekvieno opt kernelio
src/tools/memlat.cpp   atminties delsos, TLB ir pralaidumo matavimai (docs/memory.md)
scripts/bench.sh       matavimų serija -> po vieną CSV kiekvienai konfigūracijai, eilutės pridedamos prie results/csv/<config>.csv
scripts/plot.py        grafikai iš CSV (filtrai, etapų pavadinimai, žymės: žr. failo pradžią)
scripts/reports.sh     etapų ataskaitos -> reports/<stage>/plots/ (kiekvieno etapo README.md juos paaiškina)
scripts/memory.sh      atminties matavimai -> results/memory/*.csv, scripts/plot_memory.py -> docs/img/memory/
scripts/monitor.py     CPU / GPU apkrova paleidimo metu: btop / macmon ekrano nuotraukos -> docs/img/screens/, telemetrija -> results/monitor/
scripts/plot_monitor.py  branduolių apkrovos šilumos žemėlapiai -> docs/img/load/
scripts/build_report.py  ataskaita: reports/ataskaita/src/*.md -> reports/ataskaita/*.pdf (Google Chrome be lango)
scripts/report_figures.sh, scripts/plot_final.py  ataskaitos grafikai -> reports/ataskaita/img/
```

Kiekvienas `src/<tech>/<variant>.cpp` automatiškai tampa tikslu (target) `<tech>-<variant>` ir vykdomuoju failu `build/<tech>-<variant>`.
Naujas OpenCL kernelis, skirtas `opencl-base`, – tai naujas failas `src/opencl/base/<name>.cl` su kerneliais
`fill_matrix1` ir `matmul`; `opencl-opt` atveju – `src/opencl/opt/<name>.cl` su `geometry`
ir `matmul` (sąsaja (contract) aprašyta `src/opencl/common.cl`). `make bench` paleidžia kiekvieną kiekvienos programos kernelį.

## Naudojimas

```
make                       # sukompiliuoti viską
make openmp                # visi OpenMP variantai
make list                  # tikslų sąrašas

make run-openmp-base ARGS="-s 1000 -t 8 --verify"
make run-mpi-base NP=4 ARGS="-n 800 -k 600 -p 400 --verify"
make run-opencl-base ARGS="-s 2000 --kernel src/opencl/base/naive.cl"

make bench                 # visa matavimų serija (aplinkos kintamieji: SIZES, THREADS, REPEATS, VARIANTS)
make plot                  # grafikai iš results/csv/*.csv -> results/plots/ (arba IN=katalogas|failai OUT=katalogas)
make reports               # visos etapų ataskaitos -> reports/<stage>/ (arba STAGES="02 05")
make memory                # atminties delsa, TLB, pralaidumas -> docs/img/memory/ (≈ 3 min.)
make monitor               # branduolių apkrova: ekrano nuotraukos ir šilumos žemėlapiai (≈ 5 min.; brew install btop macmon tmux charmbracelet/tap/freeze)
make report                # ataskaita PDF: 0_ivadas, 1_openmp, 2_mpi, 3_opencl, 4_isvados + ataskaita_visa (arba PARTS="0_ivadas")
make compdb                # compile_commands.json IDE aplinkai
```

Programų parinktys (visoms vienodos): `-n -k -p` – dydžiai (`-s` nustato visus tris), `-x` – didžiausia matrix1 reikšmė,
`-t` – OpenMP gijų skaičius, `--seed`, `--print`, `--verify`, `--csv`.

Matrix1 reikšmės yra gryna `(seed, i, j)` funkcija, todėl su ta pačia sėkla (seed) visuose variantuose ir su bet kokiu
gijų skaičiumi gaunamos identiškos matricos; `--verify` patikrina rezultatą, palygindama jį
su nuosekliuoju skaičiavimu.
