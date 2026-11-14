title: Bendros išvados
subtitle: OpenMP, MPI ir OpenCL palyginimas
part: Išvados

# Bendros išvados

!!! fact "Svarbiausi faktai"
    - Kai n = 4000, daugyba pagreitėjo nuo 207,6 s iki 0,143 s – **≈ 1450 kartų** (visas laikas su pildymu – **≈ 940×**).
    - **125× iš to davė vien darbas su atmintimi** viena gija; lygiagretumas pridėjo ≈ 6× (8 procesoriaus branduoliai) arba ≈ 7,5× (vaizdo plokštė).
    - Lygiagretumą riboja **nuoseklioji dalis** (mažoms matricoms), **aparatūra** (dažnis, taupieji branduoliai, pralaidumas – didelėms) ir **atminties kiekis** (MPI kopijos).

## 1. Kelias: nuo 208 sekundžių iki 0,22 sekundės

![Visas vykdymo laikas, kai n = 4000: kiekvienas optimizavimo žingsnis ir kiek kartų jis greitesnis už seq-base](reports/ataskaita/img/final/kelias.png)

| n | seq-opt (1 gija) | openmp-opt2 (8 gijos) | mpi-opt (8 procesai) | opencl regblock (GPU) |
|---|------------------|-----------------------|----------------------|-----------------------|
| 4000 | 1,66 s | 0,276 s | 0,300 s | **0,220 s** |
| 16000 | 107,3 s | 17,4 s | 22,9 s | **11,1 s** |

Didžiausias žingsnis – pirmasis, ir jis nelygiagretus: `seq-base` → `seq-opt` (125×) pakeitė ne operacijų skaičių, o tai, iš kur ateina duomenys. Tik po to lygiagretumas turi prasmę.

## 2. Technologijų palyginimas

| | OpenMP | MPI | OpenCL |
|---|--------|-----|--------|
| modelis | gijos, bendra atmintis | procesai, atskira atmintis, pranešimai | tūkstančiai darbo elementų vaizdo plokštėje |
| pastangos | kelios direktyvos | duomenų paskirstymas, transliavimas, surinkimas | atskira programa vaizdo plokštei |
| pagreitėjimas | iki 6,8× | iki 6,2×, didelėms matricoms krenta | kerneliai: naive → regblock 24–29× |
| kas riboja | nuoseklioji dalis, dažnis, taupieji branduoliai | atminties kopijos | nuoseklus B pildymas (27 % laiko) |
| kada rinktis | viena mašina, bendri duomenys | klasteris, kai duomenys netelpa vienoje mašinoje | didelis vienodas skaičiavimas |

Vienoje mašinoje greičiausia vaizdo plokštė (1,2–1,5 karto greitesnė už 8–12 procesoriaus gijų), bet tik su kerneliu, pritaikytu jos atminčiai – naivus kernelis buvo 14 kartų lėtesnis už `openmp-opt2`. MPI čia moka atminties kopijomis, tačiau tas pats kodas veiktų ir klasteryje.

## 3. Ką sužinojome

![Fazių dalys, kai n = 4000: nuoseklus B pildymas užima 3 % vienos gijos laiko, bet 19–27 % greičiausių lygiagrečių variantų laiko](reports/ataskaita/img/final/fazes.png){: .narrow }

1. **Visos optimizacijos – atminties optimizacijos.** Operacijų skaičius n·k·p visur tas pats; laimėjimą lėmė tai, kiek kartų ir iš kur atnešami duomenys: procesoriuje – blokai (ir pakavimas), vaizdo plokštėje – sujungti kreipiniai, lokalioji atmintis ir registrai.
2. **Trys lygiagretumo ribos.** Nuoseklus B pildymas tuo svarbesnis, kuo greitesnė likusi dalis (3 % → 27 %; Amdalo dėsnis). Didelėms matricoms riboja aparatūra: našiųjų branduolių dažnis krinta nuo 3,5 iki ≈ 2,9 GHz, keturi taupieji branduoliai prilygsta ≈ 0,6 našiojo, pralaidumas – ≈ 122 GB/s. MPI procesų atminties kopijos sukelia atminties stygių ir sulėtina tą patį paleidimą iki 43 %.
3. **Apkrova nėra naudingas darbas.** 100 % apkrauti branduoliai gali dirbti ir 100 kartų lėčiau – lygiagretumą vertiname laiku ir pagreitėjimu.
4. **Patikrinimai būtini.** Vaizdo plokštės paleidimas, „pasibaigęs“ per 0,22 s, buvo nutrauktas kernelis; nuo tada kiekvienas paleidimas tikrinamas. Nepasiteisino MPI konvejeris (vienoje mašinoje nėra asinchroninės pažangos) ir statinis paskirstymas su taupiaisiais branduoliais.

Visus matavimus galima tyrinėti ir interaktyviai – paleisti lenktynes, peržiūrėti branduolių apkrovą, keisti Amdalo dėsnio parametrus: [dimabreeze.github.io/VU-ComputerArchitecture-Parallel-Lab](https://dimabreeze.github.io/VU-ComputerArchitecture-Parallel-Lab/) (QR kodas – titulinio puslapio apačioje).
