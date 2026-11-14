title: OpenMP: lygiagretumas bendrojoje atmintyje
subtitle: openmp-base → openmp-opt → openmp-opt2
part: 1 dalis

# 1 dalis. OpenMP

!!! fact "Svarbiausi faktai"
    - Su 8 gijomis `openmp-base` ir `openmp-opt2` apkrovos vaizde atrodo vienodai – visi našieji branduoliai 100 %, tačiau kai n = 4000, `openmp-opt2` yra **104 kartus greitesnis** (28,7 s → 0,276 s). **Apkrova nerodo naudingo darbo** (3 skyrius).
    - Geriausias `openmp-opt2` pagreitėjimas – **6,3–6,8×** (8–12 gijų, n ≥ 8000). Keturi taupieji branduoliai prie 8 našiųjų prideda **ne daugiau kaip ≈ 7 %** (n = 16000: 6,34× → 6,79×) (4 skyrius).
    - Vykdymo laikas auga kaip n³: padvigubinus n, `openmp-opt2` laikas su 8 gijomis išauga **7,5–8,4 karto** (n = 4000 → 16000) (5 skyrius).
    - Amdalo dėsnis su išmatuota nuosekliąja dalimi paaiškina mažų matricų pagreitėjimą (n = 2000: **5,2×, dėsnis – 5,6×**), bet ne didelių (n = 16000: **6,3×, dėsnis – 7,55×**) (6 skyrius).
    - Ilgesnio paleidimo su 8 gijomis metu **našiųjų branduolių dažnis nukrenta nuo 3,5 iki ≈ 2,9 GHz** – tai aparatinės įrangos nuostolis, kurio Amdalo dėsnis nemodeliuoja (3 skyrius).

OpenMP gijos dalijasi ta pačia atmintimi: visos mato tas pačias matricas, todėl duomenų siųsti nereikia, o darbą tarp gijų paskirsto kompiliatoriaus direktyvos. Šioje dalyje parodoma, kaip OpenMP programa įvykdo užduoties reikalavimus, kaip ji tobulinta per tris variantus, kaip atrodo branduolių apkrova vykdymo metu ir kaip laikas priklauso nuo gijų skaičiaus bei matricų dydžio. Matavimo sąlygos, apkrovos stebėjimo būdai ir analizės įrankiai aprašyti Bendrojoje dalyje (2, 3 ir 5 skyriai).

## 1. Užduoties reikalavimai OpenMP programoje

Visi trys variantai (`src/openmp/base.cpp`, `opt.cpp`, `opt2.cpp`) turi tą pačią sandarą: trys fazės, kiekviena matuojama atskirai `omp_get_wtime()`. Gijų skaičius nurodomas komandinėje eilutėje (`-t`) ir nustatomas `omp_set_num_threads()`.

**A pildymas – lygiagretus.** Kiekvienas elementas apskaičiuojamas iš sėklos ir indekso (`rand_1_to_x`), todėl gijoms nereikia bendro atsitiktinių skaičių generatoriaus ir rezultatas nepriklauso nuo gijų skaičiaus:

```cpp
#pragma omp parallel
{
#pragma omp single
    num_threads = omp_get_num_threads();      // kiek gijų iš tikrųjų gauta
#pragma omp for schedule(static)
    for (int i = 0; i < a.n; ++i)
        for (int j = 0; j < a.k; ++j)
            m1.at(i, j) = rand_1_to_x(a.seed, (uint64_t)i * a.k + j, a.x);
}
```

**B pildymas – nuoseklus.** Vienas generatorius (`std::mt19937_64`) eina per visus elementus iš eilės, už lygiagrečios srities ribų:

```cpp
t0 = omp_get_wtime();
fill_matrix2_sequential(m2, a.seed);          // (gen() & 1) ? 1 : -1
t.fill2 = omp_get_wtime() - t0;
```

**Daugyba – lygiagreti.** C eilutės paskirstomos gijoms; kiekviena gija rašo tik į savo eilutes, todėl sinchronizacija nereikalinga. `openmp-base` daugina vadovėline i-j-z tvarka (kaip `seq-base`):

```cpp
#pragma omp parallel for schedule(static, chunk)   // chunk = 4
for (int i = 0; i < a.n; ++i)
    for (int j = 0; j < a.p; ++j) {
        int sum = 0;
        for (int z = 0; z < a.k; ++z) sum += m1.at(i, z) * m2.at(z, j);
        m3.at(i, j) = sum;
    }
```

Naudotos OpenMP direktyvos (be jų – `omp_set_num_threads()` ir `omp_get_wtime()`):

| direktyva | kur naudojama | ką daro |
|-----------|---------------|---------|
| `#pragma omp parallel` + `single` | A pildymas | sukuria gijas; viena jų užrašo gautą gijų skaičių |
| `#pragma omp for schedule(static, 4)` | `openmp-base`: A pildymas ir daugyba | eilutės dalijamos po 4 paeiliui (round-robin) |
| `#pragma omp for schedule(static)` | `openmp-opt`, `openmp-opt2`: A pildymas; `openmp-opt`: daugyba | kiekviena gija gauna vieną ištisinį eilučių bloką |
| `#pragma omp simd` | `openmp-opt` vidinis ciklas | prašo kompiliatoriaus vektorizuoti ciklą |
| `#pragma omp parallel for schedule(dynamic, 1)` | `openmp-opt2`: daugyba | 128 eilučių blokus gijos ima iš eilės, kai baigia ankstesnį |

Kiekvieno paleidimo rezultatą galima patikrinti `--verify`: A palyginama su ta pačia formule, C – su nuosekliai apskaičiuota sandauga.

## 2. Trys variantai: base → opt → opt2

Visų trijų variantų pildymas toks pat; skiriasi tik daugyba. Kiekvienas variantas nagrinėjamas iki dydžio, ties kuriuo jį nustojome tobulinti.

| variantas | daugyba | paskirstymas | nagrinėjama iki |
|-----------|---------|--------------|-------------|
| `openmp-base` | i-j-z, B skaitoma stulpeliais | `static, 4` | n = 4000 |
| `openmp-opt` | i-k-j, B skaitoma eilutėmis, `simd` | `static` | n = 8000 |
| `openmp-opt2` | blokai + 4 × 16 mikrobranduolys registruose, `-O3 -mcpu=native` | `dynamic, 1` | n = 16000 |

### 2.1 openmp-base: atsisakyta ties n = 4000

`openmp-base` atsitrenkia į tą pačią TLB ribą kaip `seq-base` (Bendrosios dalies 4.1 skyrius): kai n = 4000, viena gija dirba 207 s, t. y. 125 kartus lėčiau nei `seq-opt`. Pagreitėjimas atrodo geras – 7,2× su 8 ir 8,9× su 12 gijų, – bet greičiausiai būtent todėl, kad kiekviena gija didžiąją laiko dalį laukia atminties: kol viena laukia, kitos irgi gali laukti, ir net lėti taupieji branduoliai padidina bendrą našumą. Lygiagretumas lėtos programos nepadaro greita: 8 gijos (28,7 s) vis tiek 17 kartų lėtesnės už vieną `seq-opt` giją (1,66 s).

### 2.2 openmp-opt: atsisakyta ties n = 8000

`openmp-opt` sukeičia du vidinius ciklus (i-k-j): A elementas `v` dauginamas iš visos B eilutės ir pridedamas prie C eilutės, todėl abi matricos skaitomos nuosekliai ir ciklą galima vektorizuoti:

```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < a.n; ++i) {
    int* c = m3.row(i);  const int* ai = m1.row(i);
    for (int z = 0; z < a.k; ++z) {
        const int v = ai[z];  const int* b = m2.row(z);
#pragma omp simd
        for (int j = 0; j < a.p; ++j) c[j] += v * b[j];
    }
}
```

Kai n = 4000, viena gija tampa 43 kartus greitesnė (207 s → 4,80 s), TLB sulėtėjimo nebėra. Vienai operacijai su viena gija tenka pastovios 0,075 ns. Tačiau **lygiagretusis plečiamumas blogėja didėjant n**: su 8 gijomis pagreitėjimas krinta nuo 5,9× (n = 2000) iki 3,9× (n = 8000), su 12 gijų – iki 3,5×.

Priežastis – atminties pralaidumas. Kiekviena C eilutė perskaito visą matricą B (64 MB, kai n = 4000), kuri netelpa į jokį podėlį. Viena `openmp-opt` gija B skaito **54 GB/s** greičiu, t. y. **93 %** išmatuoto vienos gijos pralaidumo (58 GB/s, Bendrosios dalies 4.2 skyrius). Visų gijų pralaidumas neviršija ≈ **122 GB/s**, o 8 gijoms, kai n = 4000, reikėtų ≈ 245 GB/s. Taigi `openmp-opt` jau su viena gija riboja atmintis, ne skaičiavimas, ir papildomos gijos daugiausia laukia magistralės. Prie to prisideda trys atminties operacijos (C ir B įkėlimas, C įrašymas) vienai daugybos ir sudėties operacijai ir statinis paskirstymas, kuris lėtiesiems taupiesiems branduoliams skiria tiek pat eilučių kaip našiesiems (4 skyrius).

### 2.3 openmp-opt2: blokai, registrai ir dinaminis paskirstymas

`openmp-opt2` taiko tą pačią idėją kaip `seq-opt` (Bendrosios dalies 4.3 skyrius), bet turi savo skaičiavimo branduolį (B blokai nepakuojami). Kiekvienam trūkumui – atskiras pakeitimas:

1. **Blokai podėliui (pralaidumas).** Iteracijų erdvė skaidoma blokais `BI × BK × BJ` = 128 × 256 × 128. B blokas (256 × 128 int = 128 KB) lieka podėlyje ir panaudojamas visoms 128 bloko eilutėms, užuot kiekvienai eilutei skaičius iš operatyviosios atminties.
2. **Mikrobranduolys registruose (atminties operacijos).** `micro4x16` laiko 4 × 16 C fragmentą 16 vektorinių registrų: per vieną `z` žingsnį 16 B ir 4 A skaitymai duoda 64 daugybos ir sudėties operacijas, C neskaitoma ir nerašoma iki bloko pabaigos.
3. **Dinaminis paskirstymas (nevienodi branduoliai).** Darbo vienetas – 128 C eilučių blokas; gija, baigusi bloką, ima kitą, todėl našieji branduoliai atlieka daugiau blokų nei taupieji.

```cpp
#pragma omp parallel for schedule(dynamic, 1)
for (int i0 = 0; i0 < a.n; i0 += BI)               // vienas darbo vienetas = 128 C eilučių
    for (int k0 = 0; k0 < a.k; k0 += BK)
        for (int j0 = 0; j0 < a.p; j0 += BJ)
            for (int i = i0; i + 4 <= i1; i += 4)
                for (int j = j0; j + 16 <= j1; j += 16)
                    micro4x16(m1, m2, m3, i, j, k0, k1);   // + kraštų branduoliai
// i1, k1, j1 – bloko pabaigos: min(i0 + BI, n), min(k0 + BK, k), min(j0 + BJ, p)
```

Blokų dydžiai parinkti matuojant (n = 4000, daugybos laikas): vien blokai su 8 gijomis sumažino laiką maždaug 2 kartus (0,93 → 0,46 s), mikrobranduolys su viena gija – dar ≈ 2 kartus (3,53 → 1,66 s). Galutinis rezultatas, kai n = 4000: viena gija 1,71 s (2,8 karto greičiau nei `openmp-opt`), 8 gijos – 0,276 s (4,0 karto greičiau). Vienai operacijai su viena gija tenka ≈ 0,026 ns – beveik tiek pat kaip `seq-opt` (0,025 ns).

## 3. Branduolių apkrova vykdymo metu

Apkrova užfiksuota `btop` ekrano vaizdais ir šilumos žemėlapiais (Bendrosios dalies 3 skyrius; `btop` C0–C3 – taupieji, C4–C11 – našieji branduoliai). Kiek branduolių programa užėmė vidutiniškai, apskaičiuota iš `/usr/bin/time`: procesoriaus laikas (user + sys), padalytas iš trukmės.

| paleidimas | n | trukmė | procesoriaus laikas | branduolių vidutiniškai |
|------------|---|--------|---------------------|-------------------------|
| `openmp-opt2`, 1 gija | 8000 | 13,99 s | 13,98 s | **1,00** |
| `openmp-opt2`, 8 gijos | 16000 | 17,90 s | 130,98 s | **7,32** |
| `openmp-opt2`, 12 gijų | 16000 | 17,37 s | 170,62 s | **9,82** |
| `openmp-base`, 8 gijos | 4000 | 29,83 s | 235,44 s | **7,89** |

### 3.1 Viena gija

**Lygiai vienas branduolys, bet ne tas pats.** Su viena gija `openmp-opt2` užima 1,00 branduolio, tačiau `btop` rodo dalinę kelių našiųjų branduolių apkrovą (C8–C11: 8–64 %), o šilumos žemėlapyje matyti, kaip gija kas kelias sekundes perkeliama į kitą P branduolį – kaip ir nuoseklioje programoje (Bendrosios dalies 3 skyrius). Taupieji branduoliai dirba tik su kitomis programomis.

![openmp-opt2, n = 8000, 1 gija: procesas naudoja 98,4 % vieno branduolio, apkrova išsidalijusi tarp kelių našiųjų branduolių (C8–C11)](docs/img/screens/openmp-opt2-t1-btop.png)

![openmp-opt2, n = 8000, 1 gija: gija keliauja tarp P0–P7, tačiau kiekvienu momentu dirba tik vienas branduolys (1,0 branduolio)](docs/img/load/openmp-opt2-t1.png){: .wide }

**Dirbantis branduolys visą laiką veikia 3,5 GHz.** Dažnių grafike P branduolių kreivė svyruoja apie 2,1 GHz tik todėl, kad tai visos grupės vidurkis kartu su neveikiančiais branduoliais; branduolio, kuriame tuo metu yra gija, dažnis visus 13,6 s yra 3504 MHz.

### 3.2 Aštuonios gijos

**Visi 8 našieji branduoliai – 100 %, bet vidutiniškai 7,32 branduolio, ne 8.** Daugybos metu P0–P7 užimti visiškai, o taupieji branduoliai lieka kitoms programoms. Trūkstamą dalį lemia nuoseklioji fazė: B pildymas (0,88 s) vykdomas vienu branduoliu – šilumos žemėlapyje tai raudona juosta prieš daugybą, kai dirba tik vienas P branduolys. Jei daugyba (16,7 s) užimtų lygiai 8 branduolius, o pildymas – vieną, vidurkis būtų (8 · 16,7 + 0,88) / 17,9 ≈ 7,5; likusį skirtumą, tikėtina, lemia programos pradžia ir pabaiga (atminties išskyrimas ir atlaisvinimas) bei paskutinių blokų laukimas daugybos pabaigoje. Tai Amdalo dėsnis, matomas tiesiogiai apkrovos vaizde.

![openmp-opt2, n = 16000, 8 gijos: C4–C11 (našieji) apkrauti 100 %, taupieji C0–C3 – tik kitų programų](docs/img/screens/openmp-opt2-t8-btop.png)

![openmp-opt2, n = 16000, 8 gijos: prieš daugybą – nuoseklus B pildymas vienu branduoliu; maždaug 9-ąją sekundę P branduolių dažnis nukrenta nuo 3,5 iki ≈ 2,9 GHz](docs/img/load/openmp-opt2-t8.png){: .wide }

**Našiųjų branduolių dažnis krenta nuo ≈ 3,5 iki ≈ 2,9 GHz.** Pirmąsias ≈ 8 daugybos sekundes visi 8 P branduoliai veikia 3504 MHz, procesorius naudoja ≈ 51 W. Toliau dažnis nukrenta iki ≈ 2,8–2,9 GHz ir išlieka iki pabaigos, galia – ≈ 36–38 W. Vienos gijos paleidime (3.1 skyrius) toks kritimas nepasireiškė. Greičiausiai tai lusto galios ar šilumos valdymas, kai visi našieji branduoliai ilgai apkrauti. Tai **aparatinės įrangos nuostolis, kurio Amdalo dėsnis nemodeliuoja**: maždaug pusę daugybos laiko branduolių dažnis ≈ 17 % mažesnis, nors apkrova vis dar 100 % (6 skyrius).

### 3.3 Dvylika gijų

**Užimti visi 12 branduolių, bet vidutiniškai 9,82.** Su 12 gijų `btop` rodo visus branduolius, įskaitant taupiuosius C0–C3, 100 % apkrautus. Vien nuoseklioji fazė vidurkį sumažintų tik iki ≈ 11,2; likęs ≈ 1,4 branduolio skirtumas, tikėtina, tenka kitoms programoms, kurios dirba tuose pačiuose branduoliuose (fone jos užima 1,4–1,7 branduolio, Bendrosios dalies 2 skyrius). Taupiųjų branduolių silpnumas apkrovoje nematomas – lėtas branduolys rodomas taip pat 100 % užimtas; jis matyti tik pagreitėjime (4 skyrius). Našiųjų branduolių dažnis šįkart pradeda kristi anksčiau (≈ 4-ąją sekundę) ir nusistovi ties ≈ 2,7–2,8 GHz.

![openmp-opt2, n = 16000, 12 gijų: visi 12 branduolių 100 %, procesas – 1051 %](docs/img/screens/openmp-opt2-t12-btop.png)

![openmp-opt2, n = 16000, 12 gijų: taupieji branduoliai E0–E3 dirba visą daugybos laiką (dinaminis paskirstymas), P branduolių dažnis palaipsniui krenta iki ≈ 2,7–2,8 GHz](docs/img/load/openmp-opt2-t12.png){: .wide }

### 3.4 Apkrova nėra naudingas darbas

**`openmp-base` su 8 gijomis atrodo lygiai taip pat užimtas, bet yra 104 kartus lėtesnis.** Ekrano vaizde ir šilumos žemėlapyje `openmp-base` (n = 4000, 8 gijos) nesiskiria nuo `openmp-opt2`: visi 8 našieji branduoliai 100 %, vidutiniškai 7,89 branduolio. Tačiau tam pačiam darbui `openmp-base` reikia 28,7 s, o `openmp-opt2` – 0,276 s (n = 4000, 8 gijos).

![openmp-base, n = 4000, 8 gijos: C4–C11 apkrauti 100 %, procesas – 764 % – vaizdas toks pat kaip efektyvios programos](docs/img/screens/openmp-base-t8-btop.png)

![openmp-base, n = 4000, 8 gijos: 30 s visi P branduoliai 100 % ir 3,5 GHz, bet branduoliai daugiausia laukia atminties](docs/img/load/openmp-base-t8.png){: .wide }

**Branduoliai „užimti“ laukdami.** Šio paleidimo metu P branduolių dažnis visas 30 s išlieka 3504 MHz, o procesorius naudoja tik ≈ 25 W – perpus mažiau nei `openmp-opt2` (≈ 51 W) tuo pačiu dažniu. Tikėtina, kad dauguma vykdymo įrenginių stovi, laukdami duomenų po TLB ir podėlio praleidimų (Bendrosios dalies 4.1 skyrius); tas laukimas apkrovos rodiklyje skaičiuojamas kaip darbas. Todėl apkrova parodo tik tai, kad gijos išdalytos visiems branduoliams, bet ne tai, ar branduoliai dirba naudingai.

## 4. Laikas pagal gijų skaičių

Pagreitėjimas skaičiuojamas to paties varianto atžvilgiu, S(N) = T(1) / T(N), visas laikas `total` (Bendrosios dalies 5 skyrius). Matuota su 1, 2, 4, 8 ir 12 gijų.

![openmp-opt2 vykdymo laikas pagal gijų skaičių (vertikali ašis logaritminė): laikas krinta iki 8 gijų, toliau beveik nebekinta](reports/ataskaita/img/omp/plots/openmp-opt2-time.png)

Iki 4 gijų visi dydžiai plečiasi beveik idealiai (3,3–3,7× su 4 gijomis). Nuo 4 iki 8 gijų laikas dar sutrumpėja 1,5–1,8 karto. Nuo 8 iki 12 gijų kreivės beveik horizontalios: papildomos 4 gijos tenka taupiesiems branduoliams.

### 4.1 openmp-opt: statinis paskirstymas ir atmintis

![openmp-opt pagreitėjimas: kuo didesnė matrica, tuo mažesnis pagreitėjimas; nuo n = 6000 su 12 gijų lėčiau nei su 8](reports/ataskaita/img/omp/plots/openmp-opt-speedup.png)

| n | 2000 | 3000 | 4000 | 6000 | 8000 |
|---|------|------|------|------|------|
| pagreitėjimas, 8 gijos | 5,9× | 5,5× | 4,4× | 4,5× | 3,9× |
| pagreitėjimas, 12 gijų | 6,4× | 5,0× | 4,6× | 3,7× | 3,5× |

Du reiškiniai. Pirma, pagreitėjimas mažėja didėjant n – tai atminties pralaidumo riba (2.2 skyrius). Antra, **12 gijų dažnai lėčiau nei 8**: kai n = 2500–3500 ir visada nuo n = 6000 (n = 8000: 9,81 s su 8 gijomis ir 11,18 s su 12). `schedule(static)` kiekvienai gijai skiria po n/12 eilučių, tad našieji branduoliai savo dalį baigia anksčiau ir laukia, kol lėtesni taupieji branduoliai baigs savąją. Lygiagretaus ciklo trukmę lemia lėčiausia gija.

### 4.2 openmp-opt2: plečiamumas gerėja didėjant n

`openmp-opt2` pagreitėjimo kreivės kiekvienam dydžiui pateiktos 6 skyriuje kartu su Amdalo dėsnio prognoze.

| n | 2000 | 4000 | 8000 | 10000 | 16000 |
|---|------|------|------|-------|-------|
| pagreitėjimas, 8 gijos | 5,24× | 6,18× | 6,54× | 6,75× | 6,34× |
| pagreitėjimas, 12 gijų | 5,21× | 6,38× | 6,62× | 6,64× | 6,79× |

- **Plečiamumas gerėja didėjant n** (5,2× → 6,3–6,8×), priešingai nei `openmp-opt`: blokai pašalino pralaidumo ribą, o nuoseklioji dalis mažėja (6 skyrius).
- **12 gijų nebėra pastebimai lėtesnės už 8.** Dinaminiame paskirstyme taupieji branduoliai paima mažiau blokų. Dviem atvejais 12 gijų vis dėlto truputį lėtesnės (n = 2000: 0,0428 ir 0,0425 s; n = 10000: 4,09 ir 4,02 s), bet skirtumas neviršija 2 % – matavimų sklaidos ribose.
- **Taupieji branduoliai prideda nedaug.** Daugiausia – kai n = 16000: 4 papildomi branduoliai padidina pagreitėjimą nuo 6,34× iki 6,79×, t. y. ≈ 7 %; kitiems dydžiams – nuo −2 % iki +3 %. Tai atitinka E branduolio greitį: 8 + 4 · 0,15 = 8,6 našiojo branduolio vietoj 8 (+7,5 %).

## 5. Laikas pagal matricų dydį

![Vykdymo laikas pagal matricos dydį (abi ašys logaritminės, geriausias gijų skaičius): openmp-opt2 tiesė lygiagreti n³; openmp-base ties n ≈ 3547 šokteli aukštyn](reports/ataskaita/img/omp/plots/size.png)

Abiem ašims logaritminėmis, O(n³) algoritmas atrodo kaip tiesė, kurios nuolydis 3. `openmp-opt2` taškai guli tiesėje visame intervale nuo n = 2000 iki 16000; `openmp-opt` – irgi, tik aukščiau (su geriausiu gijų skaičiumi jam reikia 2,4–4,8 karto daugiau laiko); `openmp-base` ties n ≈ 3547 šokteli aukštyn – tai TLB riba (Bendrosios dalies 4.1 skyrius).

![Tas pats laikas pagal daugybos ir sudėties operacijų skaičių n·k·p (tiesinės ašys): openmp-opt2 – tiesė, t. y. pastovi vienos operacijos kaina](reports/ataskaita/img/omp/plots/size-linear.png)

Tiesinėmis ašimis, kai X – operacijų skaičius n·k·p, pastovi vienos operacijos kaina reiškia tiesę per koordinačių pradžią. `openmp-opt2` taip ir yra: 16000³ ≈ 4,1·10¹² operacijų atliekama per 16,3 s (≈ 0,004 ns vienai operacijai su 12 gijų). `openmp-opt` kreivė statesnė, o `openmp-base` beveik vertikali.

Laikas `total`, sekundėmis (brūkšnys – variantas šiam dydžiui nenagrinėjamas, 2 skyrius):

| n | gijos | openmp-base | openmp-opt | openmp-opt2 |
|---|-------|-------------|------------|-------------|
| 2000 | 1 | 4,79 | 0,646 | 0,223 |
| 2000 | 8 | 0,645 | 0,109 | 0,0425 |
| 4000 | 1 | 207 | 4,80 | 1,71 |
| 4000 | 8 | 28,7 | 1,10 | 0,276 |
| 8000 | 1 | – | 38,6 | 13,6 |
| 8000 | 8 | – | 9,81 | 2,08 |
| 16000 | 1 | – | – | 110 |
| 16000 | 8 | – | – | 17,4 |
| 16000 | 12 | – | – | 16,3 |

Padvigubinus n, operacijų skaičius išauga 8 kartus. `openmp-opt2` su viena gija laikas išauga 7,7× (2000 → 4000), 8,0× (4000 → 8000) ir 8,1× (8000 → 16000) – beveik tiksliai n³. Su 8 gijomis – 6,5×, 7,5× ir 8,4×. Mažoms matricoms santykis mažesnis už 8, nes didelę 8 gijų laiko dalį sudaro nuoseklus B pildymas, kuris auga tik kaip n² (6 skyrius). `openmp-opt` su 8 gijomis tarp n = 4000 ir 8000 išauga 8,9 karto – daugiau nei n³, nes blogėja jo plečiamumas.

## 6. Nuoseklioji dalis ir Amdalo dėsnis

Nuoseklioji dalis f = fill2 / total išmatuota kiekvienam dydžiui (1 gija), N_eff = 8,6 su 12 gijų; formulės – Bendrosios dalies 5 skyriuje.

| n | f, % | 1/f | B pildymas 8 gijų laike, % | S(8) išmat. | S(8) Amdalas | e(8), % | S(12) išmat. | S(12) Amdalas (N_eff = 8,6) | e(12), % |
|---|------|-----|------|------|------|------|------|------|------|
| 2000 | 6,00 | 17 | 31 | 5,24 | 5,63 | 7,5 | 5,21 | 5,91 | 11,9 |
| 4000 | 3,13 | 32 | 19 | 6,18 | 6,56 | 4,2 | 6,38 | 6,94 | 8,0 |
| 8000 | 1,55 | 65 | 10 | 6,54 | 7,21 | 3,2 | 6,62 | 7,69 | 7,4 |
| 16000 | 0,84 | 119 | 5 | 6,34 | 7,55 | 3,7 | 6,79 | 8,08 | 7,0 |

![openmp-opt2: išmatuotas pagreitėjimas (ištisinės linijos) ir Amdalo dėsnis su išmatuota f ir N_eff = 8 + 0,15 · (N − 8) (punktyras): mažoms matricoms kreivės arti, didelėms – toli; po 8 gijų abi beveik plokščios](reports/ataskaita/img/omp_amdahl/plots/openmp-opt2-amdahl-hetero.png)

**Mažos matricos: Amdalo dėsnis paaiškina didžiąją dalį.** Kai n = 2000, B pildymas sudaro 6 % vienos gijos laiko, bet 31 % 8 gijų laiko – jis nesutrumpėja. Dėsnis leidžia 5,63×, išmatuota 5,24×.

**Didelės matricos: Amdalo dėsnis paaiškina mažai.** Kai n = 16000, f < 1 % ir dėsnis leidžia 7,55× su 8 gijomis, o išmatuota 6,34×. Visiems n ≥ 4000 pagreitėjimas su 8 gijomis lieka ≈ 6,2–6,75×.

**12 gijų: nevienodi branduoliai.** Klasikinis dėsnis (visi branduoliai vienodi, N = 12) prognozuotų iki 11×, išmatuota – tik 5,2–6,8×. Įvertinus, kad 4 taupieji branduoliai verti tik 0,6 našiojo (N_eff = 8,6), prognozė po 8 gijų tampa beveik plokščia – tokios pat formos kaip matavimai. Vadinasi, plokščią kreivės dalį po 8 gijų paaiškina taupiųjų branduolių silpnumas, o ne nuoseklioji dalis.

![Karpo–Flato metrika: kairėje e(N) kiekvienam dydžiui, dešinėje e su 2, 4, 8 ir 12 gijų ir išmatuota f – su 8 gijomis e laikosi ≈ 3–4 %, nors f krinta iki 0,84 %](reports/ataskaita/img/omp_amdahl/plots/openmp-opt2-karpflatt.png)

**Karpo–Flato metrika atskiria nuostolius.** Pavyzdžiui, kai n = 16000 ir 8 gijos:

    e = (1/6,34 − 1/8) / (1 − 1/8) = 3,7 %,   o išmatuota f = 0,84 %

Su 8 gijomis e ≈ 3–4 % visiems n ≥ 4000, o f per tą patį intervalą sumažėja nuo 3,13 % iki 0,84 %. Taigi e nemažėja kartu su f: skirtumas e − f – nuostoliai, nesusiję su B pildymu, – sudaro 1–3 procentinius punktus ir didėjant uždaviniui ne mažėja, o auga. Su 12 gijų e šokteli iki ≈ 7 % – tai taupieji branduoliai. Tikėtinos 8 gijų nuostolių priežastys (atskirai nematuotos): mažesnis našiųjų branduolių dažnis, kai visi jie apkrauti (3.2 skyrius), bendras L2 podėlis ir atminties pralaidumas keturioms vienos grupės gijoms, paskutinių blokų laukimas lygiagretaus ciklo pabaigoje.

Dažnio hipotezę paremia tai, kaip perteklius e − f su 8 gijomis priklauso nuo paleidimo trukmės: paleidimams iki 4 s (n ≤ 10000) jis ≈ 1–1,7 procentinio punkto, o 7–17 s paleidimams (n ≥ 12000) – ≈ 2,5–3 punktai. Be to, didžiausias pagreitėjimas su 8 gijomis gautas, kai n = 10000 (6,75×, 4,0 s), o ilgesniems paleidimams (n = 12000–16000, 7–17 s) jis nukrenta iki 6,3–6,4× – tikėtina, kad šie paleidimai jau pasiekia dažnio kritimą, kuris apkrovos matavime prasidėjo po ≈ 8 s. Tai tik hipotezė: pagrindinių matavimų metu dažnis nebuvo registruojamas, paleidimai vyko vienas po kito (lustas galėjo būti jau įkaitęs), o n = 12000 paleidimas (7,2 s) trumpesnis už ≈ 8 s, po kurių dažnis nukrito apkrovos matavime.

**Gustafsono dėsnis.** B pildymas auga kaip n², daugyba – kaip n³, todėl f mažėja nuo 6 % (n = 2000) iki 0,84 % (n = 16000), o Amdalo riba 1/f kyla nuo 17 iki 119. Todėl `openmp-opt2` pagreitėjimas didėjant n auga (5,2× → 6,3–6,8×) – iki aparatinės ribos, kurią lemia ne nuoseklioji dalis, o branduolių dažnis, bendri ištekliai ir nevienodi branduoliai. Užduotis neleidžia lygiagretinti B pildymo, bet didelėms matricoms tai ir taip duotų mažai: kai n = 16000, B pildymas sudaro tik 5 % 8 gijų laiko.

## Išvados

1. OpenMP programa įvykdo užduotį trimis direktyvomis: A pildoma `#pragma omp for`, B – nuosekliai už lygiagrečios srities ribų, daugyba – `#pragma omp parallel for` per C eilutes. Kiekviena gija rašo tik į savo eilutes, todėl sinchronizacijos nereikia.
2. Greitį labiausiai lemia ne gijų skaičius, o darbas su atmintimi. `openmp-opt2` su viena gija (1,71 s) yra 17 kartų greitesnis už `openmp-base` su 8 gijomis (28,7 s), kai n = 4000. `openmp-opt` parodė, kad programa, kuri kiekvienai operacijai skaito duomenis iš operatyviosios atminties, su 8 gijomis paspartėja tik 3,9–4,5 karto (n ≥ 4000), nes gijos dalijasi tuo pačiu ≈ 122 GB/s pralaidumu.
3. Apkrovos vaizdai patvirtina, kad gijos paskirstomos branduoliams (1,00 / 7,32 / 9,82 branduolio su 1 / 8 / 12 gijų), bet nerodo efektyvumo: `openmp-base` ir `openmp-opt2` su 8 gijomis atrodo vienodai užimti, nors skiriasi 104 kartus. Nuoseklioji B pildymo fazė apkrovos vaizde matoma kaip vieno branduolio intervalas prieš daugybą.
4. Pagal gijų skaičių: `openmp-opt2` laikas krinta beveik idealiai iki 4 gijų ir gerai iki 8 (6,2–6,75×, kai n ≥ 4000); 4 taupieji branduoliai prideda ne daugiau kaip ≈ 7 %. Taupiesiems branduoliams būtinas dinaminis paskirstymas – su statiniu (`openmp-opt`) 12 gijų dažnai lėtesnės už 8.
5. Pagal matricų dydį: `openmp-opt2` laikas auga kaip n³ (padvigubinus n – ≈ 8 kartus) visame intervale nuo n = 2000 iki 16000, o plečiamumas didėjant n gerėja, nes nuoseklioji dalis mažėja nuo 6 % iki 0,84 % (Gustafsono dėsnis).
6. Amdalo dėsnis su išmatuota f paaiškina mažų matricų pagreitėjimą, bet ne didelių: kai n = 16000, likusius ≈ 3 % nuostolių greičiausiai lemia aparatinė įranga – vykdymo metu našiųjų branduolių dažnis nukrito nuo 3,5 iki ≈ 2,9 GHz, – o plokščią kreivę po 8 gijų paaiškina nevienodi branduoliai (N_eff = 8,6).
