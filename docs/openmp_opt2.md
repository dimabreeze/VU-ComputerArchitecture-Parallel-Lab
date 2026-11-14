# OpenMP opt2: blokinė matricų daugyba

Šaltinis: [`src/openmp/opt2.cpp`](../src/openmp/opt2.cpp). Paremta
[`src/openmp/opt.cpp`](../src/openmp/opt.cpp). Pakeista tik daugybos fazė. matrix1
pildymas (lygiagretus) ir matrix2 pildymas (nuoseklus, kaip reikalauja užduotis)
yra toks pat kaip `opt`, todėl `--verify` ir duomenys visuose variantuose sutampa.

Matavimams naudotas kompiuteris: Apple M2 Pro (8 našieji (P) + 4 taupieji (E) branduoliai),
clang 18, libomp.

## 1. Kodėl `opt` buvo lėtas

`opt` jau naudoja i-k-j ciklų tvarką su `simd` vidiniu ciklu:

```cpp
for i:                       // lygiagrečiai, schedule(static)
    for z:
        v = A[i][z]
        for j: C[i][j] += v * B[z][j]
```

Išmatuotas `opt` daugybos laikas (stulpelis `mul`):

| n    | 1 gija   | 8 gijos   | pagreitėjimas |
|------|----------|-----------|---------|
| 2000 | 0,62 s   | 0,085 s   | 7,3×    |
| 4000 | 4,79 s   | 0,93 s    | 5,2×    |

Vienos gijos našumas abiem dydžiams vienodas. Tačiau lygiagretusis plečiamumas, kai
n = 4000, suprastėja. Lentelė rodo tris problemas:

1. **Atminties pralaidumas.** Kiekviena eilutė `i` perskaito visą matrix2 (64 MB, kai
   n = 4000), kuri netelpa į jokį podėlį. Visos gijos tuo pat metu iš naujo skaito
   matrix2 iš RAM, ir bendra atminties magistralė prisotinama.
2. **Per daug atminties operacijų vienai daugybos ir sudėties operacijai.** Vidinis
   ciklas kiekvienai daugybos ir sudėties operacijai atlieka `load C[j]`, `load B[j]`,
   `store C[j]`, todėl vektoriniai vykdymo įrenginiai didžiąją laiko dalį laukia įkėlimų ir
   įrašymų.
3. **Statinis paskirstymas nevienodų (heterogeninių) branduolių procesoriuje.** Su
   12 gijų `schedule(static)` lėtiesiems E branduoliams skiria tiek pat eilučių, kiek
   P branduoliams, ir visi laukia lėčiausios gijos. 12 gijų buvo lėtesnės nei 8.

## 2. Ką daro opt2

### 2.1 Blokavimas podėliui (skaidymas blokais, tiling)

Iteracijų erdvė suskaidoma blokais:

```cpp
#pragma omp parallel for schedule(dynamic, 1)
for i0 in 0..n step BI:            // vienas darbo vienetas = BI eilučių C
    for k0 in 0..k step BK:
        for j0 in 0..p step BJ:
            C[i0:i0+BI, j0:j0+BJ] += A[i0:i0+BI, k0:k0+BK] * B[k0:k0+BK, j0:j0+BJ]
```

Dabartinis matrix2 blokas (tile) (`BK x BJ` = 256 × 128 int = 128 KB) lieka L1/L2 podėlyje
ir yra pakartotinai naudojamas visoms `BI` = 128 darbo vieneto eilutėms. Be blokavimo
jis būtų iš naujo įkeliamas iš RAM kiekvienai eilutei. Tai pašalina atminties
pralaidumo ribą (1 problema).

### 2.2 Blokavimas registruose (mikrobranduolys 4 × 16)

Plytelės viduje `micro4x16` apskaičiuoja C fragmentą iš 4 eilučių × 16 stulpelių:

```cpp
int acc[4][16] = {};               // 16 NEON registrų (po 4 int)
for z in k0..k1:
    b = &B[z][j]                   // 16 gretimų int = viena podėlio eilutė
    v = {A[i][z], A[i+1][z], A[i+2][z], A[i+3][z]}
    acc[r][q] += v[r] * b[q]       // 64 daugybos ir sudėties operacijos
C[i..i+3][j..j+15] += acc          // įrašoma kartą per k bloką
```

Kiekviename `z` žingsnyje mikrobranduolys atlieka 64 daugybos ir sudėties operacijas
su 16 matricos B įkėlimų ir 4 matricos A įkėlimais, be jokių matricos C įkėlimų ar
įrašymų. `opt` reikėjo 3
atminties operacijų vienai daugybos ir sudėties operacijai, taigi tai išsprendžia
2 problemą. Mikrobranduolys parašytas paprasta C++ kalba; clang (`-O3`) jį išskleidžia
(unroll) ir laiko `acc` vektoriniuose registruose.

Kraštai, netelpantys į 4 × 16 formą, apdorojami paprastesniais skaičiavimo branduoliais:
- `kernel4`: 4 eilutės, likę stulpeliai (mažiau nei 16 `BJ` bloko gale);
- `kernel1`: viena eilutė, likusios eilutės (mažiau nei 4 `BI` bloko gale).

Teisingumas patikrintas su `--verify` nesulygiuotiems dydžiams, pavyzdžiui,
`-n 37 -k 51 -p 29`, `-n 517 -k 300 -p 333`, `-n 1001 -k 999 -p 1003`.

### 2.3 Dinaminis paskirstymas

`schedule(dynamic, 1)` eilučių blokams (po `BI` eilučių kiekviename): gija paima kitą
bloką, kai baigia ankstesnįjį. Todėl P branduoliai apdoroja daugiau blokų nei
E branduoliai, ir tai išsprendžia 3 problemą: 12 gijų dabar nebėra lėtesnės nei 8.

### 2.4 Kompiliatoriaus parinktys

Makefile tik šiai programai prideda `-O3 -mcpu=native`:

```make
build/openmp-opt2: CXXFLAGS += -O3 -mcpu=native
```

`opt2.cpp` automatiškai įtraukiamas pagal šabloną `src/<tech>/*.cpp`, todėl
`make`, `make openmp` ir `make bench` jį apima.

## 3. Blokų dydžių parinkimas

`BI`, `BK`, `BJ` yra makrokomandos, kurias galima perrašyti per `-DBI=... -DBK=... -DBJ=...`.
Jos parinktos matuojant, kai n = 4000 (`mul`, sekundės):

| versija / blokai                 | 1 gija   | 8 gijos   |
|----------------------------------|----------|-----------|
| `opt` (be blokavimo)             | 4,79     | 0,93      |
| tik blokavimas, BI=64 BK=128 BJ=2048 | 3,53     | 0,46      |
| + mikrobranduolys, BI=32 BK=256 BJ=512 | 2,10   | 0,32      |
| + mikrobranduolys, BI=64 BK=256 BJ=1024 | 1,93  | 0,28      |
| + mikrobranduolys, BI=128 BK=256 BJ=256 | 1,70  | 0,23      |
| **+ mikrobranduolys, BI=128 BK=256 BJ=128** (pasirinkta) | **1,66** | **0,22** |
| + mikrobranduolys, BI=128 BK=512 BJ=128 | 2,15  | 0,28      |

Vien blokavimas su 8 gijomis duoda apie 2× (pralaidumo problema).
Mikrobranduolys su viena gija duoda dar 2× (įkėlimų ir įrašymų problema).

## 4. Rezultatai

Iš `results/20261113-165017/` (laikas `total` = fill1 + fill2 + mul):

| n    | gijos   | opt     | opt2    | opt2 greitesnis |
|------|---------|---------|---------|----------------|
| 3000 | 1       | 2,10 s  | 0,76 s  | 2,8×           |
| 3000 | 8       | 0,36 s  | 0,13 s  | 2,8×           |
| 4000 | 1       | 4,85 s  | 1,73 s  | 2,8×           |
| 4000 | 8       | 1,13 s  | 0,28 s  | 4,1×           |
| 4000 | 12      | 1,05 s  | 0,27 s  | 3,9×           |

Paties opt2 lygiagretusis plečiamumas (`total`, 1 gija / 8 gijos): 5,8×, kai
n = 3000, ir 6,2×, kai n = 4000.

## 5. Likę apribojimai

- **Nuoseklus matrix2 pildymas (Amdalo dėsnis).** Kai n = 3000 ir 8 gijos, `fill2`
  užtrunka apie 0,03 s iš 0,13 s bendro laiko, t. y. beveik ketvirtadalį. Pagal
  užduoties apibrėžimą jis nuoseklus. Jį būtų galima pagreitinti ir neišlygiagretinant:
  generatorius `fill_matrix2_sequential` iš kiekvieno 64 bitų `mt19937_64` iškvietimo
  naudoja 1 bitą, nors galėtų naudoti visus 64. Tai pakeistų `common.cpp` ir matrix2
  duomenis visuose variantuose, todėl tai nebuvo padaryta.
- **Palyginimas su `seq-base`.** Kai n = 4000, `seq-base` yra neproporcingai lėtas
  (211 s, palyginti su 25,6 s, kai n = 3000; O(n³) prognozuoja apie 2,4×, išmatuotas
  santykis – 8,3×). Jo j-z ciklas skaito matrix2 stulpeliais 16 KB žingsniu, o tai
  sukelia podėlio ir TLB nepataikymus. Todėl opt2 pagreitėjimas `seq-base` atžvilgiu
  (apie 760× su 8 gijomis) daugiausia matuoja algoritmą ir kreipimosi į atmintį
  šabloną, o ne lygiagretumą. Lygiagretumo efektyvumui vertinti opt2 reikia lyginti su
  jo paties laiku su 1 gija. Analizė pateikta
  [tlb_cliff.md](tlb_cliff.md).
- **Kitos galimybės (neįgyvendintos):** rankomis rašytos NEON intrinsic funkcijos,
  matrix2 blokų pakavimas į ištisinius buferius, 8 × 16 mikrobranduolys, `int16`
  duomenys. Kiekviena iš jų padidina sudėtingumą arba pakeičia duomenis.
