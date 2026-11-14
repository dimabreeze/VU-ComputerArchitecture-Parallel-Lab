# MPI opt: greitas skaičiavimo branduolys, blokinis paskirstymas, (nebūtinas) konvejerinis transliavimas

Šaltinis: [`src/mpi/opt.cpp`](../src/mpi/opt.cpp), tikslas `mpi-opt`. Remiasi
[`src/mpi/base.cpp`](../src/mpi/base.cpp) (`mpi-base`, žr. toliau) ir išlaiko užduoties semantiką:

- matrix1 pildoma lygiagrečiai, kiekvienas procesas pildo savo eilutes;
- matrix2 pildoma nuosekliai rango 0 procese;
- daugyba vykdoma lygiagrečiai.

`--verify` sėkmingai praeina su 1, 3 ir 8 procesais nesulygiuotiems dydžiams, įskaitant
atvejį, kai procesų daugiau nei eilučių (`-n 2 -k 7 -p 33` su 8 procesais).

Kompiuteris: Apple M2 Pro (8 našieji (P) ir 4 taupieji (E) branduoliai), Open MPI, vienas mazgas
(procesai keičiasi duomenimis per bendrai naudojamą atmintį).

## 0. Atspirties taškas: mpi-base

`mpi-base` ([`src/mpi/base.cpp`](../src/mpi/base.cpp)) yra kurso šablonas
(„MPI program sample“, [`docs/Reference/mpi_sample.md`](Reference/mpi_sample.md)),
pataisytas, kad veiktų teisingai, ir šiek tiek pritaikytas šiai platformai. Daugybos algoritmas ir
darbo paskirstymas paimti iš šablono: eilutės skirstomos cikliškai
(`i = rank, rank + size, ...`), kiekviena eilutė skaičiuojama i-j-z ciklu ir siunčiama
rangui 0 su `MPI_Send` / `MPI_Recv`.

Pataisyta, palyginti su šablonu:

- matricos generuojamos pagal užduoties sąlygą, ir visuose procesuose gaunami tie patys duomenys.
  Šablone kiekvienas procesas generavo savo matricas su `rand()`,
  todėl procesai daugino skirtingus duomenis. Dabar matrix1 pildoma lygiagrečiai
  (kiekvienas procesas pildo savo eilutes), o matrix2 pildoma nuosekliai rango 0 procese ir
  išsiunčiama visiems procesams su `MPI_Bcast`;
- pranešimų žymės: šablonas eilutei i naudojo žymę 10+i, kuri, esant dideliam n, viršija `MPI_TAG_UB`.
  Naudojama po vieną žymę kiekvienam srautui; MPI išlaiko pranešimų tarp tos pačios
  procesų poros tvarką;
- perteklinis `}` prieš `MPI_Finalize`; ataskaitą spausdina tik rangas 0.

Pridėta platformai: komandinės eilutės argumentai, deterministiniai generatoriai (kad
veiktų `--verify`), rezultato tikrinimas ir CSV ataskaita.

Dėl transliavimo `mpi-base` fazė `fill2` yra „nuoseklus pildymas + `MPI_Bcast`“.
Todėl ji neįtraukta į fill2 pasiskirstymo grafiką (`fill2-distribution.png`),
kuriame rodomas tik grynas nuoseklus pildymas.

## 1. Pakeitimai, palyginti su mpi-base

### 1.1 Greitas skaičiavimo branduolys (pagrindinis laimėjimas)

`mpi-base` daugina i-j-z ciklu, kuris skaito matrix2 stulpeliais ir, kai n > 3500, atsiremia į
TLB ribą (žr. [tlb_cliff.md](tlb_cliff.md)). `mpi-opt` naudoja
`gemm_add` iš [`include/gemm.hpp`](../include/gemm.hpp): blokavimą (skaidymą blokais), matrix2
pakavimą ir 4 × 16 mikrobranduolį registruose. Tai tas pats skaičiavimo branduolys kaip `seq-opt`,
žr. [seq_opt.md](seq_opt.md). Jis perkeltas į bendrą antraštės failą, kad abi programos
naudotų vieną realizaciją. Kompiliuojama su `-O3 -mcpu=native` (Makefile).

### 1.2 Blokinis eilučių paskirstymas ir MPI_Gatherv

`mpi-base` eilutes skirsto cikliškai (`i = rank, rank + size, ...`). Tada rangas 0
gauna n atskirų `MPI_Recv` pranešimų, o kiekvienas procesas išskiria atmintį pilnoms n × k ir
n × p matricoms.

`mpi-opt` kiekvienam procesui skiria vieną ištisinį eilučių bloką:
`n / size` eilučių ir dar vieną papildomą eilutę pirmiesiems `n % size` rangams:

- procesas saugo tik savo matrix1 ir matrix3 eilutes. Rangas 0 saugo pilnas
  matricas, nes jam priklauso pirmasis blokas ir jam vis tiek reikia viso rezultato;
- matrix3 surenkama **vienu** `MPI_Gatherv`. Rangas 0 naudoja `MPI_IN_PLACE`, todėl
  jo paties blokas nekopijuojamas;
- mikrobranduolys dirba su ištisiniu eilučių bloku.

### 1.3 matrix2 transliavimas: nebūtinas konvejeris

matrix2 reikia kiekvienam procesui. Galimi du režimai:

- **numatytasis (`KC = 0`):** vienas visų k eilučių transliavimas, po to daugyba;
- **konvejeris (`-DKC=<rows>`):** matrix2 siunčiama dalimis po `KC` eilučių
  neblokuojančiu `MPI_Ibcast`. Visos dalys inicijuojamos iš karto. Procesas laukia
  dalies c, daugina su ja (`C += A[:, chunk] * B[chunk, :]`), o tuo metu
  gali būti perduodamos dalys c+1... Tarp stulpelių blokų jis kviečia `MPI_Test`, kad
  jų perdavimas progresuotų.

Abiem režimais transliavimo laikas įskaitomas į daugybos fazę (`mul`). `mpi-base`
jis buvo matrix2 fazės (`fill2`) dalis, todėl `mpi-opt` fazė `fill2` yra
tik nuoseklus generavimas.

## 2. Kodėl konvejeris numatytuoju atveju išjungtas

Konvejeris buvo realizuotas ir išmatuotas (n = 4000, `mul`, geriausias iš 3 paleidimų):

| procesai | viena dalis (numatytasis) | KC = 1024 | KC = 512 | KC = 256 | KC = 128 |
|----------|---------------------------|-----------|----------|----------|----------|
| 1        | **1,56 s**                | 1,61      | 1,61     | 1,72     | 1,90     |
| 4        | **0,417 s**               | 0,432     | 0,441    | 0,472    | 0,517    |
| 8        | **0,243 s**               | 0,256     | 0,263    | 0,289    | 0,326    |
| 12       | **0,246 s**               | 0,262     | 0,285    | 0,296    | 0,347    |

Šiuo atveju konvejeris yra 5–40 % lėtesnis dėl trijų priežasčių.

1. **Mažai ką slėpti.** Paprastas 64 MB (4000 × 4000 int) `MPI_Bcast`
   trunka 2,4 ms su 2 procesais, 5,5 ms su 4, 15 ms su 8 ir 24 ms su 12. Tai sudaro
   6–10 % daugybos laiko.
2. **Viename kompiuteryje nėra tikro persidengimo.** Viename mazge transliavimas yra atminties
   kopijavimas, kurį per bendrai naudojamą atmintį atlieka patys procesai. Nėra atskiro įrenginio,
   kuris perkeltų duomenis, kol CPU skaičiuoja, kitaip nei klasteryje su RDMA tinklo plokšte.
   Open MPI neblokuojančias kolektyvines operacijas varo į priekį tik MPI kvietimų viduje.
   `mul` fazės profiliavimas su 8 procesais tai patvirtina: laikas, praleistas
   `MPI_Wait`, su konvejeriu ne sumažėjo, o *išaugo*:

   | 8 procesai, n = 4000 | laukimas MPI_Wait | skaičiavimas | surinkimas | mul iš viso |
   |----------------------|-------------------|--------------|------------|-------------|
   | viena dalis          | 0,014–0,024 s     | 0,207–0,216 s | 0–0,019 s | 0,243 s |
   | KC = 512             | 0,021–0,040 s     | 0,216–0,219 s | 0–0,023 s | 0,265 s |

   Dažnesnis `MPI_Testall` kvietimas visoms laukiančioms dalims (kas 128 arba 32 stulpelius)
   nepadėjo: `mul` liko 0,253–0,279 s.
3. **Mažos dalys kenkia skaičiavimo branduoliui.** Kiekviena dalis reiškia atskirą proceso
   matrix3 bloko perėjimą. Dalys, mažesnės už skaičiavimo branduolio k bloką
   (`GEMM_BK` = 512), be to, sutrumpina mikrobranduolio vidinį ciklą. Todėl
   KC = 128 yra lėčiausias.

**Išvada:** komunikacijos ir skaičiavimo sujungimas į konvejerį atsiperka, kai
perdavimas brangus ir vyksta asinchroniškai, kaip klasteryje su tinklu.
Viename bendrai naudojamos atminties kompiuteryje jis tik prideda pridėtinių sąnaudų nieko nepaslėpdamas.
Kodas paliktas programoje (`-DKC=512` jį įjungia) kaip demonstracija ir paleidimams
tikrame klasteryje.

## 3. Rezultatai

`total` laikas, n = 4000 (`mul` apima transliavimą ir surinkimą):

| procesai | mpi-base | mpi-opt | kiek kartų greičiau |
|----------|----------|---------|---------------------|
| 1        | 212,0 s  | 1,66 s  | 128×                |
| 2        | 106,8 s  | 0,89 s  | 120×                |
| 4        | 55,3 s   | 0,48 s  | 115×                |
| 8        | 29,3 s   | 0,31 s  | 95×                 |
| 12       | 25,8 s   | 0,38 s  | 68×                 |

Paties mpi-opt plečiamumas (`mul`, 1 procesas / p procesų): 1,93× su 2, 3,8× su 4,
6,4× su 8.

**12 procesų veikia lėčiau nei 8.** Eilutės padalytos po lygiai, bet 4 iš 12
procesų veikia lėtesniuose taupiuosiuose (E) branduoliuose. Kiekvienas procesas `MPI_Gatherv`
laukia lėčiausiojo. Profilis tai rodo kaip didelę skaičiavimo laiko sklaidą
tarp rangų: 0,168–0,219 s su 12 procesų, palyginti su 0,207–0,216 s su 8.
Apkrovos balansavimas (svertinis padalijimas arba dinaminis paskirstymas pagal schemą
„vadovas–vykdytojai“ (master-worker)) sąmoningai nerealizuotas. Efektas paliktas kaip
nevienodų (heterogeninių) branduolių demonstracija.

## 4. Likę apribojimai

- **Transliavimas ir surinkimas**, po 64 MB kiekvienas: su 8 procesais apie 15 + 15 ms,
  daugiau nei 10 % `mul`. Viename kompiuteryje vienintelis būdas jų išvengti būtų MPI-3
  bendrai naudojamos atminties langai (`MPI_Win_allocate_shared`), tačiau šio būdo atsisakyta, kad
  duomenys ir toliau būtų perduodami pranešimais.
- **Nuoseklus matrix2 pildymas** rango 0 procese (`fill2` ≈ 0,056 s, kai n = 4000) yra
  užduoties dalis ir pagal Amdalo dėsnį riboja bendrą laiką, kaip ir `openmp-opt2`.
