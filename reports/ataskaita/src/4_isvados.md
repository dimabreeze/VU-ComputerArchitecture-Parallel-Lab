title: Bendros išvados
subtitle: OpenMP, MPI ir OpenCL palyginimas
part: Išvados

# Bendros išvados

!!! fact "Svarbiausi faktai"
    - Kai n = 4000, daugyba pagreitėjo nuo 207,6 s iki 0,143 s – **≈ 1450 kartų**, visas laikas su matricų pildymu – nuo 207,6 s iki 0,220 s (**≈ 940×**) (1 skyrius).
    - **125× iš to davė vien darbas su atmintimi** viena gija; lygiagretumas pridėjo dar ≈ 6× (8 procesoriaus branduoliai) arba ≈ 7,5× (vaizdo plokštė) (1 skyrius).
    - Greičiausia – **vaizdo plokštė**, bet tik po trijų optimizavimo žingsnių: naivus kernelis buvo **14 kartų lėtesnis** už OpenMP su 8 gijomis (2 skyrius).
    - Lygiagretumą riboja trys skirtingi dalykai: **nuoseklioji dalis** (Amdalo dėsnis, mažoms matricoms), **aparatūra** (dažnio kritimas, taupieji branduoliai, pralaidumas – didelėms) ir **atminties kiekis** (MPI kopijos) (3 skyrius).
    - Procesoriaus apkrova **nerodo naudingo darbo**: 100 % apkrauti branduoliai gali dirbti ir 100 kartų lėčiau (3 skyrius).

## 1. Kelias: nuo 208 sekundžių iki 0,22 sekundės

![Visas vykdymo laikas, kai n = 4000: kiekvienas optimizavimo žingsnis ir kiek kartų jis greitesnis už seq-base](reports/ataskaita/img/final/kelias.png)

| n | seq-base | seq-opt (1 gija) | openmp-opt2 (8 gijos) | mpi-opt (8 procesai) | opencl regblock (GPU) |
|---|----------|------------------|-----------------------|----------------------|-----------------------|
| 4000 | 207,6 s | 1,66 s | 0,276 s | 0,300 s | **0,220 s** |
| 8000 | – | 12,9 s | 2,08 s | 2,13 s | **1,47 s** |
| 16000 | – | 107,3 s | 17,4 s | 22,9 s | **11,1 s** |

Didžiausias žingsnis – pirmasis, ir jis visai nelygiagretus. `seq-base` → `seq-opt` (125×) pakeitė ne operacijų skaičių, o tai, iš kur ateina duomenys: blokai lieka podėlyje, supakuota B skaitoma nuosekliai, rezultatas kaupiamas registruose. Tik po to lygiagretumas turi prasmę: 8 procesoriaus branduoliai pagreitina dar ≈ 6×, vaizdo plokštė – ≈ 7,5×.

![Visų geriausių variantų laikas pagal matricos dydį (abi ašys logaritminės): tiesės lygiagrečios n³, t. y. vienos operacijos kaina nuo dydžio nebepriklauso](reports/ataskaita/img/final/plots/size.png)

## 2. Technologijų palyginimas

| | OpenMP | MPI | OpenCL |
|---|--------|-----|--------|
| modelis | gijos, bendra atmintis | procesai, atskira atmintis, pranešimai | tūkstančiai darbo elementų vaizdo plokštėje |
| ką teko keisti kode | kelios direktyvos | duomenų paskirstymas, transliavimas, surinkimas | atskira programa vaizdo plokštei, pagrindinė programa, buferiai |
| geriausias laikas, n = 4000 | 0,267 s (12 gijų) | 0,300 s (8 procesai) | **0,220 s** |
| geriausias laikas, n = 16000 | 16,3 s (12 gijų) | 22,9 s (8 procesai) | **11,1 s** |
| pagreitėjimas, palyginti su 1 gija (procesu) | iki 6,8× | iki 6,2×, didelėms matricoms krenta | – (fiksuoti 19 skaičiavimo vienetų) |
| kas riboja | nuoseklioji dalis, dažnis, taupieji branduoliai | atminties kopijos, taupieji branduoliai | nuoseklus B pildymas procesoriuje (27 %), sveikųjų skaičių aritmetika |
| kada rinktis | viena mašina, bendri duomenys | kai duomenys netelpa vienoje mašinoje (klasteris) | didelis vienodas skaičiavimas, kai yra galinga vaizdo plokštė |

Vienoje mašinoje MPI neturi pranašumų prieš OpenMP: ta pati skaičiavimo idėja (blokai ir 4 × 16 mikrobranduolys registruose), bet kiekvienas procesas laiko savo matricos B kopiją, o pranešimai keliauja per atmintį. MPI pranašumas – tas pats kodas veiktų ir klasteryje, kur OpenMP neveikia.

Vaizdo plokštė už 8–12 procesoriaus gijų greitesnė 1,2 karto (n = 4000) – 1,5 karto (n = 16000), bet tik su kerneliu, pritaikytu jos atminčiai. Naivus kernelis (3,95 s, kai n = 4000) buvo 14 kartų lėtesnis už `openmp-opt2`: be sujungtų kreipinių, lokaliosios atminties ir registrų vaizdo plokštė daugiausia laukia duomenų.

## 3. Kas lemia greitį

### 3.1 Visos optimizacijos – atminties optimizacijos

Operacijų skaičius n·k·p visuose variantuose tas pats. Kiekvienas didelis laimėjimas – sprendimas, iš kur ir kiek kartų atnešami duomenys:

- **procesoriuje:** stulpelio skaitymas kerta puslapių ribas (TLB, n > 3547); i-k-j tvarka tai išsprendžia, bet atsiremia į pralaidumą (viena gija naudoja 93 % jo); blokavimas (`seq-opt` ir `mpi-opt` – dar ir B pakavimas) sumažina srautą iš atminties šimtus kartų;
- **vaizdo plokštėje:** sujungti kreipiniai (7×), lokalioji atmintis (1,45×) ir registrai (2,6×) – ta pati idėja, kitais žodžiais.

### 3.2 Trys lygiagretumo ribos

![Fazių dalys, kai n = 4000: nuoseklus B pildymas užima 3 % vienos gijos laiko, bet 19–27 % greičiausių lygiagrečių variantų laiko](reports/ataskaita/img/final/fazes.png)

1. **Nuoseklioji dalis (Amdalo dėsnis).** B pildymas pagal užduotį nuoseklus. Kuo greitesnė lygiagreti dalis, tuo didesnę laiko dalį jis užima: 3 % vienos gijos, 19 % su 8 gijomis ir 27 % vaizdo plokštėje. Mažoms matricoms tai paaiškina beveik visą pagreitėjimo ribą. Didėjant matricoms nuoseklioji dalis mažėja kaip 1/n (Gustafsono dėsnis).
2. **Aparatūra.** Didelėms matricoms nuoseklioji dalis mažesnė nei 1 %, bet pagreitėjimas su 8 gijomis lieka ≈ 6,5×. Apkrovos stebėjimas parodė priežastis: kai dirba visi našieji branduoliai, jų dažnis krinta nuo ≈ 3,5 iki ≈ 2,9 GHz; taupusis branduolys atlieka tik ≈ 0,15 našiojo darbo (12 gijų verta ≈ 8,6 našiojo branduolio); bendras atminties pralaidumas – ≈ 122 GB/s.
3. **Atminties kiekis.** MPI procesai laiko atskiras matricos B kopijas. Kai n = 14000–16000, jos kartu su kitomis programomis užpildo 16 GB: sistema pradeda glausti atmintį ir naudoti keitimo sritį, taupieji branduoliai užimami sistemos darbu, o tas pats paleidimas sulėtėja iki 43 % priklausomai nuo laisvos atminties.

### 3.3 Apkrova nėra naudingas darbas

`seq-base` ir `seq-opt` užima lygiai po vieną branduolį, `openmp-base` ir `openmp-opt2` su 8 gijomis – po ≈ 8 branduolius. Apkrovos vaizdai atrodo vienodai, tačiau optimizuoti variantai tame pačiame laike atlieka šimtą kartų daugiau operacijų. Lėtas branduolys, laukiantis duomenų iš atminties, operacinei sistemai atrodo lygiai taip pat užimtas kaip skaičiuojantis. Todėl lygiagretumą vertiname laiku ir pagreitėjimu, o ne apkrova.

### 3.4 Patikrinimai būtini

Vienas vaizdo plokštės paleidimas „baigėsi“ per 0,22 s – greičiau, nei leidžia vaizdo plokštės didžiausias našumas: kernelis buvo nutrauktas, o neteisingas laikas pateko į rezultatus. Nuo tada kiekvienas paleidimas tikrina kernelio būseną ir atsitiktinai perskaičiuoja 256 rezultato elementus; visi variantai tikrinami `--verify` su nesulygiuotais dydžiais. Greitis be teisingumo patikrinimo nieko nereiškia.

## 4. Kas nepasiteisino

- **MPI konvejeris** (`MPI_Ibcast` dalimis, kad perdavimas persidengtų su skaičiavimu) vienoje mašinoje nepagreitino: Open MPI neturi asinchroninės pažangos per bendrai naudojamą atmintį, o transliavimas sudaro tik ≈ 6 % daugybos laiko. Jis paliktas išjungtas, kaip demonstracija klasteriui.
- **Statinis paskirstymas** su taupiaisiais branduoliais: `openmp-opt` su 12 gijų lėtesnis nei su 8. Dinaminis paskirstymas (`openmp-opt2`) šią problemą išsprendė, o MPI variante apkrovos balansavimas sąmoningai nepridėtas, kad efektas būtų matomas.
- **12 gijų vietoj 8** beveik nieko neduoda: keturi taupieji branduoliai šiame uždavinyje prilygsta ≈ 0,6 našiojo.

## 5. Išvados

1. Matricų daugybos greitį lemia atminties hierarchija: tas pats skaičiavimas, kitaip išdėstytas atmintyje, viena gija tapo 125 kartus greitesnis.
2. Visos trys technologijos leidžia išnaudoti lygiagretumą, bet tik kartu su gera atminties prieiga: geriausi variantai yra 690–940 kartų greitesni už pradinį (n = 4000).
3. Vienoje mašinoje greičiausia vaizdo plokštė (OpenCL), po jos – OpenMP; MPI čia moka atminties kopijomis, bet išliktų tinkamas klasteriui.
4. Lygiagretumo ribą lemia ne tik Amdalo dėsnis: didelėms matricoms ją nustato aparatūra – dažnis, nevienodi branduoliai, pralaidumas ir atminties kiekis. Šiuos efektus pavyko ne tik numanyti, bet ir išmatuoti.
5. Matavimai turi būti tikrinami: rezultato teisingumas (`--verify`, atrankinis patikrinimas), pakartojimai ir medianos, matavimo sąlygų (fono programų) aprašymas.
