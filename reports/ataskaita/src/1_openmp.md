title: OpenMP: lygiagretumas bendrojoje atmintyje
subtitle: openmp-base → openmp-opt → openmp-opt2
part: 1 dalis

# 1 dalis. OpenMP

!!! fact "Svarbiausi faktai"
    - Su 8 gijomis `openmp-base` ir `openmp-opt2` apkrovos vaizde atrodo vienodai – visi našieji branduoliai 100 %, tačiau kai n = 4000, `openmp-opt2` yra **104 kartus greitesnis** (28,7 s → 0,276 s): **apkrova nerodo naudingo darbo** (3 skyrius).
    - Geriausias `openmp-opt2` pagreitėjimas – **6,3–6,8×**; 4 taupieji branduoliai prie 8 našiųjų prideda **ne daugiau kaip ≈ 7 %** (4 skyrius).
    - Laikas auga kaip n³: padvigubinus n, `openmp-opt2` laikas su viena gija išauga **7,7–8,1 karto** visame intervale nuo n = 2000 iki 16000 (5 skyrius).
    - Mažų matricų pagreitėjimą paaiškina Amdalo dėsnis, didelių – ne: ilgo paleidimo metu **našiųjų branduolių dažnis nukrenta nuo 3,5 iki ≈ 2,9 GHz** (3 ir 6 skyriai).

OpenMP gijos dalijasi ta pačia atmintimi, todėl duomenų siųsti nereikia – darbą tarp gijų paskirsto kompiliatoriaus direktyvos. Matavimo sąlygos, apkrovos stebėjimas ir analizės įrankiai aprašyti Bendrosios dalies 2, 3 ir 5 skyriuose.

## 1. Užduoties reikalavimai OpenMP programoje

Visi trys variantai (`src/openmp/base.cpp`, `opt.cpp`, `opt2.cpp`) turi tas pačias tris fazes, kiekviena matuojama `omp_get_wtime()`; gijų skaičius nustatomas `omp_set_num_threads()`.

**A pildymas – lygiagretus.** Elementas apskaičiuojamas iš sėklos ir indekso, todėl rezultatas nepriklauso nuo gijų skaičiaus:

```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < n; ++i)
    for (int j = 0; j < k; ++j) A[i][j] = rand_1_to_x(seed, i * k + j, x);
```

**B pildymas – nuoseklus**, vienu `std::mt19937_64` generatoriumi, už lygiagrečios srities ribų:

```cpp
fill_matrix2_sequential(m2, seed);     // B[i][j] = (gen() & 1) ? 1 : -1
```

**Daugyba – lygiagreti** per C eilutes; kiekviena gija rašo tik į savo eilutes, todėl sinchronizacijos nereikia. `openmp-opt2` darbo vienetas – 128 C eilučių blokas:

```cpp
#pragma omp parallel for schedule(dynamic, 1)
for (int i0 = 0; i0 < n; i0 += BI)     // BI = 128 eilučių
    ...                                // blokai BK × BJ, mikrobranduolys micro4x16
```

## 2. Trys variantai: base → opt → opt2

Pildymas visuose variantuose tas pats, skiriasi tik daugyba.

| variantas | daugyba | paskirstymas | n = 4000, 1 gija | n = 4000, 8 gijos |
|-----------|---------|--------------|------------------|-------------------|
| `openmp-base` | i-j-z, B skaitoma stulpeliais | `static, 4` | 207 s | 28,7 s |
| `openmp-opt` | i-k-j, B skaitoma eilutėmis, `simd` | `static` | 4,80 s | 1,10 s |
| `openmp-opt2` | 128 KB B blokai + 4 × 16 mikrobranduolys registruose | `dynamic, 1` | 1,71 s | 0,276 s |

**`openmp-base` atsisakyta ties n = 4000.** Jis atsitrenkia į TLB ribą (Bendrosios dalies 4.1 skyrius): net su 8 gijomis jis 17 kartų lėtesnis už vieną `seq-opt` giją (1,66 s).

**`openmp-opt` atsisakyta ties n = 8000**, nes jo plečiamumas blogėja didėjant n: su 8 gijomis pagreitėjimas krinta nuo 5,9× (n = 2000) iki 3,9× (n = 8000). Kiekviena C eilutė perskaito visą B iš operatyviosios atminties: viena gija tai daro 54 GB/s greičiu – jau 93 % išmatuoto vienos gijos pralaidumo (58 GB/s), o visos gijos kartu neviršija ≈ 122 GB/s (Bendrosios dalies 4.2 skyrius).

**`openmp-opt2`** laiko B bloką podėlyje ir panaudoja jį 128 eilučių, o 4 × 16 C fragmentą kaupia registruose. Jo skaičiavimo branduolys `micro4x16` savas (ne `include/gemm.hpp`), B blokai nepakuojami. Dinaminis paskirstymas leidžia našiesiems branduoliams paimti daugiau blokų nei taupiesiems. Vienai operacijai su viena gija tenka ≈ 0,026 ns – beveik kaip `seq-opt` (0,025 ns).

Išsamiau: `docs/openmp_opt2.md`, `reports/04-openmp/README.md`.

## 3. Branduolių apkrova vykdymo metu

Kiek branduolių programa užėmė vidutiniškai, apskaičiuota iš `/usr/bin/time`: procesoriaus laikas (user + sys), padalytas iš trukmės (Bendrosios dalies 3 skyrius; `btop` C0–C3 – taupieji, C4–C11 – našieji branduoliai).

![openmp-opt2, n = 8000, 1 gija: procesas naudoja 98,4 % vieno branduolio, apkrova išsidalijusi tarp kelių našiųjų branduolių](docs/img/screens/openmp-opt2-t1-btop.png)
![openmp-opt2, n = 8000, 1 gija: gija keliauja tarp P0–P7, tačiau kiekvienu momentu dirba tik vienas branduolys](docs/img/load/openmp-opt2-t1.png)

**1 gija – lygiai 1,00 branduolio.** Gija perkeliama tarp našiųjų branduolių, kaip ir nuoseklioje programoje. Dirbantis branduolys visą laiką veikia 3,5 GHz (dažnių grafike P kreivė žemesnė, nes tai visos grupės vidurkis su neveikiančiais branduoliais).

![openmp-opt2, n = 16000, 8 gijos: C4–C11 (našieji) apkrauti 100 %, taupieji C0–C3 – tik kitų programų](docs/img/screens/openmp-opt2-t8-btop.png)
![openmp-opt2, n = 16000, 8 gijos: prieš daugybą – nuoseklus B pildymas vienu branduoliu; ≈ 9-ąją sekundę P branduolių dažnis nukrenta nuo 3,5 iki ≈ 2,9 GHz](docs/img/load/openmp-opt2-t8.png)

**8 gijos – 7,32 branduolio, ne 8: Amdalo dėsnis apkrovos vaizde.** Daugybos metu visi 8 našieji branduoliai užimti 100 %, bet prieš ją 0,88 s B pildomas vienu branduoliu; vien tai sumažina vidurkį iki ≈ 7,5, likusį skirtumą, tikėtina, lemia programos pradžia, pabaiga ir paskutinių blokų laukimas.

**Našiųjų branduolių dažnis krenta nuo 3,5 iki ≈ 2,9 GHz.** Pirmąsias ≈ 8 daugybos sekundes P branduoliai veikia 3504 MHz (≈ 51 W), toliau – ≈ 2,8–2,9 GHz (≈ 36–38 W); vienos gijos paleidime to nėra. Greičiausiai tai lusto galios ar šilumos valdymas – aparatinės įrangos nuostolis, kurio Amdalo dėsnis nemodeliuoja, nors apkrova lieka 100 %.

**12 gijų – 9,82 branduolio iš 12.** Taupieji branduoliai `btop` rodomi 100 % užimti, tačiau pagreitėjimas, palyginti su 8 gijomis, padidėja tik nuo 6,34× iki 6,79× (n = 16000): lėtas branduolys apkrovoje atrodo lygiai taip pat užimtas kaip greitas.

**Apkrova ≠ naudingas darbas.** `openmp-base` su 8 gijomis (n = 4000) užima 7,89 branduolio ir atrodo 100 % užimtas lygiai kaip `openmp-opt2`, bet yra 104 kartus lėtesnis; jo branduoliai visą laiką veikia 3,5 GHz, tačiau naudoja tik ≈ 25 W – tikėtina, didžiąją laiko dalį laukia duomenų iš atminties.

## 4. Laikas pagal gijų skaičių

Pagreitėjimas S(N) = T(1) / T(N) skaičiuojamas to paties varianto atžvilgiu (Bendrosios dalies 5 skyrius); matuota su 1, 2, 4, 8 ir 12 gijų.

![openmp-opt2 laikas pagal gijų skaičių (logaritminė ašis): laikas krinta iki 8 gijų, toliau beveik nekinta](reports/ataskaita/img/omp/plots/openmp-opt2-time.png)
![openmp-opt2 pagreitėjimas: didesnės matricos plečiasi geriau; po 8 gijų kreivės beveik plokščios](reports/ataskaita/img/omp/plots/openmp-opt2-speedup.png)

Pagreitėjimas (opt – `openmp-opt`, opt2 – `openmp-opt2`):

| n | opt, 8 gijos | opt, 12 gijų | opt2, 8 gijos | opt2, 12 gijų |
|---|------|------|------|------|
| 2000 | 5,9× | 6,4× | 5,2× | 5,2× |
| 4000 | 4,4× | 4,6× | 6,2× | 6,4× |
| 8000 | 3,9× | 3,5× | 6,5× | 6,6× |
| 16000 | – | – | 6,3× | 6,8× |

**Iki 4 gijų laikas krinta beveik proporcingai, nuo 4 iki 8 – dar 1,5–1,8 karto**, o po 8 gijų beveik nebekinta: papildomos gijos tenka taupiesiems branduoliams, kurie šiame uždavinyje atlieka tik ≈ 0,15 našiojo darbo (Bendrosios dalies 5 skyrius).

**`openmp-opt` su 12 gijų dažnai lėtesnis nei su 8** (kai n = 2500–3500 ir visada nuo n = 6000): statinis paskirstymas taupiesiems branduoliams skiria tiek pat eilučių kaip našiesiems, ir visi laukia lėčiausios gijos.

**`openmp-opt2` su dinaminiu paskirstymu 12 gijų nebėra pastebimai lėtesnės už 8** (blogiausiu atveju lėtesnės 2 %), o didžiausias taupiųjų branduolių indėlis – ≈ 7 % (n = 16000). Priešingai nei `openmp-opt`, jo plečiamumas didėjant n **gerėja** (5,2× → 6,3–6,8×).

Išsamiau: `reports/04-openmp/README.md`.

## 5. Laikas pagal matricų dydį

![Laikas pagal matricos dydį (abi ašys logaritminės, geriausias gijų skaičius): openmp-opt2 tiesė lygiagreti n³; openmp-base ties n ≈ 3547 šokteli aukštyn](reports/ataskaita/img/omp/plots/size.png)

| n | 2000 | 4000 | 8000 | 16000 |
|---|------|------|------|-------|
| `openmp-opt2`, 1 gija | 0,223 s | 1,71 s | 13,6 s | 110 s |
| `openmp-opt2`, 8 gijos | 0,0425 s | 0,276 s | 2,08 s | 17,4 s |

Abiem ašims logaritminėmis, O(n³) algoritmas yra tiesė su nuolydžiu 3. `openmp-opt2` taškai guli joje visame intervale: padvigubinus n, laikas su viena gija išauga 7,7–8,1 karto. Su 8 gijomis mažoms matricoms santykis mažesnis (6,5×, kai n = 2000 → 4000), nes nuoseklus B pildymas auga tik kaip n². `openmp-opt` tiesė lygiagreti, bet 2,4–4,8 karto aukščiau, o `openmp-base` ties n ≈ 3547 lūžta dėl TLB (Bendrosios dalies 4.1 skyrius).

## 6. Amdalo dėsnis

![openmp-opt2: išmatuotas pagreitėjimas (ištisinės linijos) ir Amdalo dėsnis su išmatuota f ir N_eff = 8 + 0,15 · (N − 8) (punktyras): mažoms matricoms kreivės arti, didelėms – toli](reports/ataskaita/img/omp_amdahl/plots/openmp-opt2-amdahl-hetero.png)

Kai n = 2000, nuoseklus B pildymas sudaro f = 6 % vienos gijos laiko ir dėsnis leidžia 5,6× su 8 gijomis – išmatuota 5,2×, taigi mažų matricų ribą paaiškina nuoseklioji dalis. Didėjant n, f mažėja iki 0,84 % (Gustafsono dėsnis), tačiau kai n = 16000, išmatuota tik 6,3× vietoj leidžiamų 7,55×: Karpo–Flato metrika su 8 gijomis lieka ≈ 3–4 %, nors f < 1 %. Šį likutį, tikėtina, lemia aparatinė įranga – nukritęs našiųjų branduolių dažnis (3 skyrius), bendri podėliai ir atminties pralaidumas, o plokščią kreivę po 8 gijų paaiškina taupieji branduoliai (N_eff = 8,6).

Išsamiau: `docs/amdahl.md`.

## Išvados

1. OpenMP programa įvykdo užduotį direktyvomis: A pildoma `#pragma omp parallel for`, B – nuosekliai, daugyba – `#pragma omp parallel for` per C eilutes be sinchronizacijos.
2. Greitį labiausiai lemia darbas su atmintimi, ne gijų skaičius: `openmp-opt2` su viena gija (1,71 s) 17 kartų greitesnis už `openmp-base` su 8 gijomis (28,7 s), o `openmp-opt` su 8 gijomis paspartėja tik 3,9–4,5 karto (n ≥ 4000), nes gijos dalijasi ≈ 122 GB/s pralaidumu.
3. Apkrovos vaizdai rodo, kaip gijos paskirstytos (1,00 / 7,32 / 9,82 branduolio su 1 / 8 / 12 gijų), bet ne efektyvumą: vienodai „užimti“ variantai skiriasi 104 kartus, o lėti taupieji branduoliai atrodo taip pat užimti kaip našieji.
4. `openmp-opt2` laikas auga kaip n³ ir krinta iki 8 gijų (6,2–6,8×, kai n ≥ 4000); 4 taupieji branduoliai prideda ne daugiau kaip ≈ 7 % ir tik su dinaminiu paskirstymu nelėtina programos.
5. Amdalo dėsnis paaiškina mažų matricų pagreitėjimą; didelių matricų nuostolį lemia aparatinė įranga – visiems našiesiems branduoliams dirbant, jų dažnis nukrito nuo 3,5 iki ≈ 2,9 GHz.
