# seq-opt: greita vienos gijos daugyba

Šaltinis: [`src/seq/opt.cpp`](../src/seq/opt.cpp), kūrimo tikslas (target) `seq-opt`. Viena gija, be
OpenMP. Sukuria tas pačias matricas ir tą patį rezultatą kaip `seq-base`, o `--verify`
sėkmingai praeina nesulygiuotiems dydžiams (`-n 37 -k 51 -p 29`, `-n 1001 -k 999 -p 1003`, `-n 3 -k 1 -p 40`).
Tikslas – parodyti, kokią pagreitėjimo dalį galima gauti visiškai be lygiagretumo.

Kompiuteris: Apple M2 Pro, clang 18, `-O3 -mcpu=native` (nustatyta Makefile tik šiam
tikslui).

## Metodai

1. **Ciklų tvarka ir blokavimas (skaidymas blokais, GotoBLAS stiliumi).**
   `j0 (BJ) -> k0 (BK) -> i0 (BI) -> 16-column panel -> MR rows`.
   matrix2 blokas (tile) `BK x BJ` panaudojamas visoms matrix1 eilutėms ir tik tada
   pereinama prie kito. matrix1 blokas `BI x BK` lieka podėlyje, kol dauginamas su
   kiekviena matrix2 bloko juosta (panel).
2. **matrix2 pakavimas (packing).** Kiekvienas blokas nukopijuojamas į ištisinį
   buferį 16 stulpelių juostomis. Už matricos krašto esantys stulpeliai užpildomi
   nuliais. Tada mikrobranduolys skaito matrix2 griežtai nuosekliai, po 64 baitus per
   žingsnį. Taip išvengiama ėjimo stulpeliu puslapio dydžio žingsniu, dėl kurio
   `seq-base` atsiremia į TLB ribą (žr. [tlb_cliff.md](tlb_cliff.md)). Be to, nebereikia
   atskiro skaičiavimo branduolio stulpelių likučiui.
3. **Registrų mikrobranduolys 4 × 16.** 16 NEON registrų laiko matrix3 fragmentą
   4 × 16 per visą k bloką. Kiekviename žingsnyje mikrobranduolys įkelia 16 matrix2
   int reikšmių, transliuoja 4 matrix1 int reikšmes ir įvykdo 16 vektorinių `mla`
   instrukcijų (64 daugybos ir sudėties operacijos). Mikrobranduolys parašytas
   paprasta C++ kalba; clang jį išskleidžia (unroll) ir vektorizuoja.

Parametrai (makrokomandos, kurias galima perrašyti per `-D`): `MR=4`, `BI=64`, `BK=512`, `BJ=512`.

## Spąstai, rasti derinant parametrus

Pirmoji versija rezultatą įrašydavo ciklu su kintama riba stulpelių likučiui:

```cpp
for (int q = 0; q < w; ++q) c[q] += acc[r][q];   // w nėra kompiliavimo laiko konstanta
```

Dėl kintamo indekso clang laikė `acc` atmintyje. Skaičiavimai vis tiek vyko
registruose, tačiau **kiekvienoje** z ciklo iteracijoje visi 16 kaupiklių registrų
buvo išstumiami į steką (8 `stp` instrukcijos „karštajame“ cikle). Pataisymas
nukopijuoja `acc` į laikiną masyvą naudodamas tik konstantinius indeksus. Po
pataisymo karštajame cikle nebeliko įrašymų į atmintį, o mikrobranduolys tapo 2×
greitesnis:

| versija (n = 4000, mul) | laikas |
|-------------------------|--------|
| be eilučių blokavimo    | 3,4 s  |
| + BI blokavimas         | 3,3 s  |
| + pataisytas likutis (be išstūmimų į steką) | 1,6 s |

Sugeneruotam kodui patikrinti: `clang++ -O3 -mcpu=native -Iinclude -S src/seq/opt.cpp`,
tada vidiniame cikle ieškoti `mla.4s` ir `stp`.

## Rezultatai (mul, viena gija)

| n    | seq-base | seq-opt | greitesnis   |
|------|----------|---------|--------------|
| 2000 | 4,78 s   | 0,218 s | 22×          |
| 3000 | 25,6 s   | 0,707 s | 36×          |
| 3500 | 49,9 s   | 1,07 s  | 47×          |
| 3600 | 84,5 s   | 1,21 s  | 70×          |
| 4000 | 211 s    | 1,67 s  | 126×         |

`seq-opt` sugaišta apie 0,026 ns vienai daugybos ir sudėties operacijai esant bet
kokiam dydžiui, o staigaus sulėtėjimo po n = 3500 nėra. Jo vienos gijos greitis
sutampa su `openmp-opt2` su 1 gija (1,66–1,73 s, kai n = 4000). Tai patvirtina, kad
opt2 pranašumas prieš bazinius variantus kyla iš kreipimosi į atmintį šablono, o
gijos prie to prideda tik lygiagretųjį plečiamumą.

Likęs atotrūkis iki aparatinės įrangos piko: 64·10⁹ / 1,67 s ≈ 38·10⁹ int32 daugybos
ir sudėties operacijų/s. Apytikslis vieno M2 našiojo (P) branduolio pikas yra apie
56·10⁹ (4 vektoriniai konvejeriai × 4 elementai (lanes) × ~3,5 GHz), taigi seq-opt
pasiekia apie 70 % jo.
