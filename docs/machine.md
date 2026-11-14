# Kompiuteris, kuriame atlikti visi matavimai

Visi šio darbo matavimai atlikti viename kompiuteryje. Jo atminties hierarchijos
matavimai (delsa, TLB, pralaidumas) – [memory.md](memory.md).

## Aparatinė įranga

| parametras | reikšmė | šaltinis |
|------------|---------|----------|
| kompiuteris | MacBook Pro (Mac14,9) | `system_profiler SPHardwareDataType` |
| lustas | Apple M2 Pro | `sysctl machdep.cpu.brand_string` |
| CPU branduoliai | 12: 8 našieji (P) + 4 taupieji (E) | `sysctl hw.perflevel0.physicalcpu`, `hw.perflevel1.physicalcpu` |
| P branduolių grupės | 2 grupės po 4 branduolius, kiekviena su savo L2 | `sysctl hw.perflevel0.cpusperl2` |
| P branduolio dažnis | iki ≈ 3,5 GHz | Apple specifikacija (macOS jo nepateikia) |
| GPU | 19 branduolių, Metal 3 | `system_profiler SPDisplaysDataType` |
| operatyvioji atmintis | 16 GB LPDDR5, bendroji CPU ir GPU atmintis | `sysctl hw.memsize` |
| atminties pralaidumas | 200 GB/s (Apple skelbiama); CPU išmatuota iki 122 GB/s | [memory.md](memory.md) |

## Podėliai, puslapiai ir TLB

| lygis | P branduolys | E branduolys | šaltinis |
|-------|--------------|--------------|----------|
| podėlio eilutė | **128 B** (32 `int`) | 128 B | `sysctl hw.cachelinesize` |
| L1 duomenų podėlis | **128 KB** kiekvienam branduoliui | 64 KB | `hw.perflevel0.l1dcachesize`, `hw.perflevel1.l1dcachesize` |
| L1 instrukcijų podėlis | 192 KB | – | `hw.perflevel0.l1icachesize` |
| L2 podėlis | **16 MB** bendras 4 P branduoliams | 4 MB bendras 4 E branduoliams | `hw.perflevel0.l2cachesize`, `hw.perflevel1.l2cachesize` |
| SLC (sistemos lygio podėlis, atlieka L3 vaidmenį) | bendras visam lustui, ≈ 24 MB | | trečiųjų šalių įvertis; macOS nepateikia (`hw.l3cachesize` nėra) |
| puslapis | **16 KB** (4096 `int`) | | `sysctl hw.pagesize` |
| L1 duomenų TLB | ≈ 256–384 įrašai | | išmatuota, [memory.md](memory.md) |
| L2 TLB | **≈ 3072 įrašai** × 16 KB ≈ 48 MB aprėptis | | išmatuota, [memory.md](memory.md) |

Išmatuota delsa (vienas atsitiktinis skaitymas, 1 gija, P branduolys):
L1 – 0,88 ns (≈ 3 taktai), L2 – 5,6 ns (≈ 20 taktų), operatyvioji atmintis – ≈ 117 ns
(≈ 400 taktų). Taktai apskaičiuoti pagal 3,5 GHz dažnį. Smulkiau – [memory.md](memory.md).

### Ką tai reiškia matricoms

Kai n = 4000, kiekviena matrica (n × n `int`) užima 64 MB = 4096 puslapių. Tai daugiau nei
L2 (16 MB), SLC (≈ 24 MB) ir TLB aprėptis (48 MB). Viena matricos eilutė – 16 000 B, t. y.
beveik vienas puslapis ir 125 podėlio eilutės. Todėl ėjimas matricos B stulpeliu kiekvienu
žingsniu paliečia naują podėlio eilutę ir beveik visada naują puslapį
([tlb_cliff.md](tlb_cliff.md)).

## Programinė įranga

| komponentas | versija |
|-------------|---------|
| operacinė sistema | macOS 15.1.1 (24B91), arm64 |
| C++ kompiliatorius | Homebrew clang 18.1.8 (`llvm@18`), `-std=c++17 -O2`; `seq-opt`, `openmp-opt2`, `mpi-opt` – `-O3 -mcpu=native` |
| OpenMP | Homebrew `libomp` 23.1.5 |
| MPI | Open MPI 5.0.11 (`mpicxx`, `mpirun`) |
| OpenCL | Apple OpenCL 1.2 (macOS dalis; pasenusi, bet veikia) |
| grafikai | Python 3.14, matplotlib 3.11 (`.venv`) |

## Kaip pakartoti

```sh
sysctl hw.cachelinesize hw.pagesize hw.memsize \
       hw.perflevel0.physicalcpu hw.perflevel0.l1dcachesize hw.perflevel0.l2cachesize hw.perflevel0.cpusperl2 \
       hw.perflevel1.physicalcpu hw.perflevel1.l1dcachesize hw.perflevel1.l2cachesize
system_profiler SPHardwareDataType SPDisplaysDataType
make memory          # delsos, TLB ir pralaidumo matavimai -> docs/memory.md
```
