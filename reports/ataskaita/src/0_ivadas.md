title: Lygiagretusis matricų daugybos programavimas
subtitle: OpenMP, MPI ir OpenCL kompiuteryje Apple M2 Pro
part: Bendroji dalis

# Įvadas

!!! fact "Svarbiausi faktai"
    - Visi matavimai atlikti **įprastai naudojamame kompiuteryje**: prieš paleidimus kitos programos jau užėmė 7–12 GB iš 16 GB operatyviosios atminties ir apkrovė 1,4–1,7 procesoriaus branduolio (2 skyrius). Rezultatai – pakartotinių paleidimų medianos.
    - Vienos gijos programos **macOS nelaiko viename branduolyje**: gija nuolat perkeliama tarp 8 našiųjų branduolių, nors iš viso užima lygiai vieną (3 skyrius).
    - Paprasta daugyba (i-j-z) **staiga sulėtėja trigubai, kai n > 3547**: kiekvienas B stulpelio žingsnis atneša naują podėlio eilutę, iš kurios panaudojama tik 3 %, ir 97,7 % atvejų – naują puslapį; kai puslapių daugiau nei telpa TLB (≈ 3072 įrašai, išmatuota), adresų vertimas tampa brangus (4 skyrius).
    - Vien geriau naudojant atmintį **viena gija tampa 125 kartus greitesnė** (n = 4000: 208 s → 1,66 s) – dar prieš bet kokį lygiagretumą. Procesoriaus apkrova abiem atvejais ta pati – vienas branduolys 100 %: **apkrova nerodo, ar branduolys dirba naudingai** (4 skyrius).

## 1. Užduotis ir ataskaitos sandara

Laboratorinio darbo užduotis – trimis technologijomis (OpenMP, MPI ir OpenCL) parašyti programą, kuri:

1. sukuria matricas A (n × k), B (k × p) ir C (n × p);
2. A užpildo atsitiktiniais sveikaisiais skaičiais iš intervalo [1, x] – **lygiagrečiai**;
3. B užpildo atsitiktinai reikšmėmis 1 ir −1 – **nuosekliai**;
4. apskaičiuoja C = A × B – **lygiagrečiai**;

ir išanalizuoti, kaip vykdymo laikas priklauso nuo **(a) naudojamų branduolių skaičiaus** ir **(b) matricų dydžio**, pateikiant procesoriaus branduolių apkrovos vaizdus programos vykdymo metu.

Ataskaita sudaryta iš savarankiškų dalių. Ši bendroji dalis aprašo visoms dalims bendrus dalykus; kiekviena tyrimo dalis remiasi tik ja, todėl gali būti pateikiama atskirai arba kartu su kitomis.

| dalis | turinys |
|-------|---------|
| **Bendroji dalis** (šis dokumentas) | užduotis, kompiuteris ir matavimo sąlygos, apkrovos stebėjimas, nuoseklus atskaitos taškas, atminties hierarchija, analizės įrankiai |
| **1 dalis. OpenMP** | lygiagretumas bendrojoje atmintyje: `openmp-base` → `openmp-opt` → `openmp-opt2`, gijų skaičius, dydis, Amdalo dėsnis, taupieji branduoliai |
| **2 dalis. MPI** | procesai su atskira atmintimi: `mpi-base` → `mpi-opt`, procesų skaičius, dydis, atminties stygius |
| **3 dalis. OpenCL** | skaičiavimai vaizdo plokštėje: `naive` → `coalesced` → `tiled` → `regblock`, dydis, GPU apkrova |
| **Bendros išvados** | visų technologijų palyginimas ir apibendrinimas |

Visas kodas, matavimų duomenys ir šios ataskaitos šaltiniai – saugykloje (nuoroda titulinio puslapio apačioje). Kiekvieną grafiką ir lentelę galima atkurti viena komanda (`make bench`, `make reports`, `make memory`, `make monitor`, `make report`).

## 2. Kompiuteris ir matavimo sąlygos

!!! conditions "Matavimo sąlygos: kompiuteris nebuvo „švarus“"
    Matavimai atlikti įprastai naudojamame nešiojamajame kompiuteryje, kuriame veikė ir kitos programos (naršyklė, programavimo aplinka, pokalbių programos). Galutinio apkrovos stebėjimo metu (3 skyrius) **prieš paleidžiant programą buvo užimta 7,4–9,7 GB RAM ir 3,0–3,3 GB keitimo srities (swap)**, ankstesniuose paleidimuose – iki 11,7 GB RAM, o kitos programos vidutiniškai apkrovė **1,4–1,7 branduolio**. Pagrindiniai matavimai (`make bench`) atlikti per kelias dienas panašiomis sąlygomis.

    Kaip tai paveikė rezultatus:

    - kiekviena konfigūracija matuota 3–7 kartus, naudojama **mediana**, ne vidurkis – atsitiktiniai fono trikdžiai jos beveik nekeičia;
    - fonas labiausiai jaučiamas ten, kur trūksta atminties. Tas pats `mpi-opt` paleidimas (n = 14000, 8 procesai) truko **23,3 s, kai laisvos atminties buvo ≈ 5 GB, ir 16,3 s, kai jos buvo ≈ 6,5 GB** (2 dalis);
    - mažiausi laikai (iki ≈ 0,05 s) jautriausi fonui – jų santykinė paklaida didžiausia.

### 2.1 Aparatinė įranga

| parametras | reikšmė |
|------------|---------|
| kompiuteris | MacBook Pro (Mac14,9), Apple M2 Pro |
| procesorius | 12 branduolių: **8 našieji (P)** – dvi grupės po 4, iki ≈ 3,5 GHz – ir **4 taupieji (E)** |
| podėliai | L1d 128 KB kiekvienam P branduoliui; L2 16 MB kiekvienai 4 P branduolių grupei; sistemos podėlis (SLC) ≈ 24 MB (įvertis) |
| atmintis | 16 GB LPDDR5, bendra procesoriui ir vaizdo plokštei; podėlio eilutė 128 B, puslapis 16 KB |
| vaizdo plokštė | 19 branduolių, Metal 3 |

### 2.2 Programinė įranga ir įrankiai

| paskirtis | įrankiai ir versijos |
|-----------|----------------------|
| operacinė sistema | macOS 15.1.1 |
| kompiliavimas ir vykdymas | Homebrew clang 18.1.8 (`-O2`, optimizuotiems variantams `-O3 -mcpu=native`); OpenMP – `libomp` 23.1.5; MPI – Open MPI 5.0.11; OpenCL – Apple OpenCL 1.2; GNU Make; Homebrew 7.0 |
| matavimai | `scripts/bench.sh` (matavimų serija), savo programa `memlat` (atminties delsa, TLB, pralaidumas), `/usr/bin/time` (procesoriaus laikas) |
| apkrovos stebėjimas | btop 1.4.8, macmon 0.11.0, tmux 3.8, freeze 0.2.2 (3 skyrius) |
| grafikai ir analizė | Python 3.14, matplotlib 3.11, numpy 2.5 |
| ataskaita | Markdown (python-markdown 3.11), Google Chrome 157 (spausdinimas į PDF) |
| pristatymai | pptxgenjs 4.0 (Node.js 22), Microsoft PowerPoint 16.115 |
| versijų kontrolė | git 2.47, GitHub |
| QR kodas | qrcodebanana.com |
| DI asistentas | Claude Code (Anthropic), modeliai Claude Fable ir Claude Opus – pagalba rašant kodą ir analizuojant rezultatus |

!!! note "Dirbtinio intelekto naudojimas"
    Rašant programas, matavimų ir grafikų scenarijus bei analizuojant rezultatus naudotas DI asistentas Claude Code (Anthropic). **Visi ataskaitoje pateikti skaičiai gauti realiais matavimais šiame kompiuteryje** – jų neapskaičiavo ir neįvertino DI. Matavimų duomenys (`results/`) ir scenarijai, kuriais gauti visi grafikai ir lentelės, yra saugykloje, todėl kiekvieną rezultatą galima patikrinti ir atkurti.

### 2.3 Kaip matuojama

- **Fazės matuojamos atskirai:** A pildymas (`fill1`), B pildymas (`fill2`), daugyba (`mul`) ir bendras laikas (`total`), monotoniniu laikrodžiu (`std::chrono::steady_clock`, OpenMP – `omp_get_wtime`, MPI – `MPI_Wtime`).
- **Visi variantai generuoja tuos pačius duomenis:** kiekvienas matricos elementas yra deterministinė funkcija nuo sėklos ir indeksų, nepriklausomai nuo gijų ar procesų skaičiaus. Todėl laikus galima tiesiogiai palyginti, o rezultatą – patikrinti (`--verify` palygina su nuosekliu skaičiavimu).
- **Matavimų serija** (`scripts/bench.sh`) paleidžia kiekvieną variantą su kiekvienu dydžiu ir gijų (procesų) skaičiumi kelis kartus; kiekvienas paleidimas – atskira CSV eilutė. Grafikuose naudojama mediana.
- **Matricos kvadratinės:** n = k = p, nuo 2000 iki 16000.

## 3. Kaip stebime branduolių apkrovą

Užduotis reikalauja parodyti branduolių apkrovą programos vykdymo metu. Naudojame du papildomus būdus.

**Ekrano vaizdai.** Programa `btop` rodo kiekvieno branduolio apkrovą ir istorijos grafiką, `macmon` – Apple Silicon branduolių grupių, vaizdo plokštės apkrovą ir energijos sąnaudas. Abi veikia terminale; jų vaizdas užfiksuojamas daugybos viduryje ir paverčiamas paveikslu (`make monitor`). Procesų sąraše paliekama tik matuojama programa.

`btop` branduolius numeruoja C0–C11: **C0–C3 yra taupieji (E), C4–C11 – našieji (P) branduoliai.**

**Šilumos žemėlapiai.** Kas 100 ms įrašoma kiekvieno branduolio ir vaizdo plokštės apkrova bei dažnis (`macmon`, be administratoriaus teisių). Žemėlapyje eilutės – branduoliai (GPU, P7…P0, E3…E0), ašis X – laikas nuo programos paleidimo, šviesesnė spalva – didesnė apkrova. Viršuje pažymėtos programos fazės, apačioje – dažniai ir atminties naudojimas. Antraštėje nurodyta, kiek branduolių programa užėmė vidutiniškai: jos procesoriaus laikas (user + sys), padalytas iš trukmės.

![seq-base (n = 3000) vykdymo metu: btop rodo vieną procesą, kuris naudoja ≈ 100 % (vieną branduolį), bet apkrova pasiskirsčiusi per kelis P branduolius](docs/img/screens/seq-base-btop.png)

![seq-base šilumos žemėlapis: viena gija kas kelias sekundes perkeliama į kitą P branduolį; iš viso programa užima lygiai 1,0 branduolio](docs/img/load/seq-base.png){: .wide }

**Ką tai rodo.** Vienos gijos programa nebūna „prikalta“ prie vieno branduolio: macOS planuoklė ją nuolat perkelia tarp 8 našiųjų branduolių (P0–P7). Todėl ekrano vaizde vienos gijos programa gali atrodyti kaip kelių branduolių dalinė apkrova, nors iš tikrųjų užima lygiai vieną branduolį. Tolesnėse dalyse tai svarbu skaitant 1 gijos vaizdus.

## 4. Bendras atskaitos taškas: nuosekli daugyba ir atmintis

Visų trijų dalių pradžia – ta pati nuosekli programa. Ji parodo, kad greitį pirmiausia lemia ne skaičiavimas, o tai, iš kurio atminties lygio ateina duomenys.

### 4.1 seq-base ir staigus sulėtėjimas ties n ≈ 3547

`seq-base` daugina vadovėline tvarka i-j-z: kiekvienam C elementui A eilutė dauginama iš B **stulpelio**. Matricos atmintyje saugomos eilutėmis, todėl gretimi stulpelio elementai nutolę 4·p baitų (n = 4000 – 16 000 B, beveik visas 16 KB puslapis). Kiekvienas žingsnis patenka į naują podėlio eilutę ir beveik visada į naują puslapį.

![Matricos atmintyje (n = 4000): kiekviena – ištisinis 64 MB blokas; B eilutė (16 000 B) trumpesnė už puslapį (16 384 B), todėl eilučių ir puslapių ribos „slenka“; viena podėlio eilutė – 32 skaičiai](docs/img/slides/atmintis-isdestymas.png)

Ką kerta kiekvienas stulpelio žingsnis, galima suskaičiuoti tiksliai:

- **podėlio eilutės ribą – 100 % žingsnių:** šuolis (16 000 B) daug ilgesnis už eilutę (128 B), todėl kiekvienas žingsnis atneša naują eilutę;
- **puslapio ribą – 97,7 % žingsnių:** tame pačiame puslapyje lieka tik tie žingsniai, kurių poslinkis puslapyje mažesnis nei 384 B (384 / 16 384 = 2,3 %);
- **iš atneštos eilutės panaudojama tik 3 %** (4 B iš 128 B). Likę 31 skaičius reikalingi kitiems stulpeliams j+1…j+31 – jei eilutė dar bus podėlyje. Visas stulpelis – 4000 eilučių × 128 B ≈ 500 KB: netelpa L1 (128 KB), bet telpa L2 (16 MB), todėl iki n ≈ 3500 eilutės randamos L2.

![Vienas B stulpelio žingsnis kerta podėlio eilutės ribą visada, puslapio ribą – 97,7 % atvejų; kai puslapių daugiau nei telpa TLB, n > 3547](docs/img/slides/ribu-kirtimas.png)

Kol visi vieno stulpelio puslapiai telpa TLB (adresų vertimo buferyje), vertimai randami greitai. Stulpelis paliečia ≈ 4·n² / 16384 puslapių; kai jų daugiau nei ≈ 3072 TLB įrašai, kiekvienas kreipinys reikalauja puslapių lentelės peržiūros:

    4 · n² / 16384 > 3072   ⇒   n > 3547

![seq-base laikas pagal operacijų skaičių n·k·p: kol vienos operacijos kaina pastovi, taškai guli tiesėje; ties n ≈ 3547 tiesė lūžta (seq-base, openmp-base ir mpi-base, 1 gija)](reports/ataskaita/img/tlb/plots/size-linear.png)

| n | 2000 | 3000 | 3500 | 3550 | 3600 | 3700 | 4000 |
|---|------|------|------|------|------|------|------|
| ns vienai daugybos ir sudėties operacijai | 0,58 | 0,89 | 1,10 | 1,26 | 1,81 | 3,18 | 3,24 |

Visi trys baziniai variantai (`seq-base`, `openmp-base` ir `mpi-base` su 1 gija) guli ant tos pačios kreivės: tai atminties, ne lygiagretumo savybė.

### 4.2 Išmatuota atminties hierarchija

Kad šie paaiškinimai remtųsi ne prielaidomis, atminties savybes išmatavome atskira programa (`make memory`): rodyklių sekimu atsitiktine tvarka matuojama vieno skaitymo delsa, kai duomenų kiekis didėja, ir kai paliečiama vis daugiau puslapių.

![Vieno skaitymo kelias: adresas verčiamas TLB, tada ieškoma L1, L2, SLC ir operatyviojoje atmintyje; po kiekvienu lygiu – išmatuota jo kaina](docs/img/slides/skaitymo-kelias.png)

![Vieno atsitiktinio skaitymo delsa pagal duomenų kiekį: kiekvienas atminties lygis – atskiras laiptelis](docs/img/memory/lat.png)

| lygis | dydis | delsa |
|-------|-------|-------|
| L1 podėlis | 128 KB | **0,88 ns** (≈ 3 taktai) |
| L2 podėlis | 16 MB | **5,6 ns** (≈ 20 taktų) |
| operatyvioji atmintis | 16 GB | **≈ 117 ns** (≈ 400 taktų) – ≈ 130 kartų lėčiau už L1 |

![TLB: po vieną eilutę kiekviename 16 KB puslapyje; šuolis tarp 3072 ir 3328 puslapių – L2 TLB riba](docs/img/memory/tlb.png)

Kai paliečiama daugiau nei **3072 puslapių** (48 MB), vieno skaitymo laikas šokteli nuo 7,7 iki **16,3 ns**. Taip tiesiogiai patvirtinta TLB talpa, kuria remiasi n ≈ 3547 riba.

Nuoseklaus skaitymo pralaidumas: **58 GB/s viena gija** ir ne daugiau kaip **≈ 122 GB/s** visomis gijomis (Apple nurodo 200 GB/s visam lustui kartu su vaizdo plokšte).

### 4.3 seq-opt: tas pats skaičiavimas, kitas darbas su atmintimi

`seq-opt` atlieka lygiai tas pačias operacijas, bet:

- **skaido matricas blokais**, kad B blokas (512 × 512 = 1 MB) liktų L2 podėlyje ir būtų panaudotas visoms A eilutėms;
- **pakuoja** B bloką į ištisinį buferį – mikrobranduolys jį skaito tik nuosekliai, TLB problema išnyksta;
- **skaičiuoja registruose:** 4 × 16 C fragmentas laikomas 16 vektorinių registrų, per žingsnį 20 skaitymų duoda 64 operacijas.

![seq-base ir seq-opt laikas pagal matricos dydį (abi ašys logaritminės): seq-opt tiesė lygiagreti n³ – vienos operacijos kaina pastovi nuo n = 2000 iki 16000](reports/ataskaita/img/seqopt/plots/size.png)

| n | seq-base | seq-opt | kiek kartų greičiau |
|---|----------|---------|---------------------|
| 2000 | 4,65 s | 0,224 s | 21× |
| 4000 | 208 s | 1,66 s | **125×** |
| 16000 | – | 107 s | |

**Apkrova nerodo efektyvumo.** Procesoriaus apkrovos vaizde `seq-opt` atrodo lygiai taip pat kaip `seq-base`: vienas procesas, ≈ 100 % vieno branduolio, gija perkeliama tarp našiųjų branduolių. Abu užima lygiai 1,0 branduolio, tačiau `seq-opt` per tą patį laiką atlieka 125 kartus daugiau operacijų. `seq-base` branduolys didžiąją laiko dalį laukia duomenų iš atminties, o ši laukimo trukmė apkrovos rodiklyje matoma kaip „darbas“.

![seq-opt (n = 8000) vykdymo metu: tas pats vaizdas kaip seq-base – vienas procesas, ≈ 100 % vieno branduolio](docs/img/screens/seq-opt-btop.png)

![seq-opt šilumos žemėlapis: viena gija, 1,0 branduolio – kaip ir seq-base, nors skaičiuoja 125 kartus greičiau](docs/img/load/seq-opt.png){: .wide }

**Kodėl 125 kartus.** Blokavimas keičia ne operacijų skaičių, o tai, kiek kartų kiekvienas skaičius atnešamas iš operatyviosios atminties. Ciklų tvarka j0 → k0 → pakuoti B → visos A eilutės: kiekvienas B blokas iš atminties perskaitomas **vieną kartą** ir panaudojamas visoms 4000 A eilučių. Kai n = 4000, iš atminties perskaitoma ≈ 1,6 GB (B – kartą, A – 8 kartus, C – 8 kartus skaitant ir rašant). Palyginimui: tvarka i-k-j kiekvienai A eilutei perskaito visą B – iki 4·n³ = 256 GB. Tai **≈ 160 kartų mažiau** baitų (įvertis pagal ciklų struktūrą).

![Blokavimas ir pakavimas: B blokas (1 MB) supakuojamas į ištisinį buferį ir lieka L2; operatyviosios atminties srautas sumažėja nuo 256 GB iki ≈ 1,6 GB](docs/img/slides/blokavimas-srautas.png)

Tai patogu išreikšti **aritmetiniu intensyvumu** – kiek daugybos ir sudėties operacijų tenka vienam iš atminties atneštam baitui. Vienas branduolys atlieka ≈ 40·10⁹ operacijų per sekundę, o atmintis vienai gijai tiekia 58 GB/s, todėl kompiuterio **balansas ≈ 0,7 operacijos baitui**. Algoritmas su mažesniu intensyvumu laukia atminties, su didesniu – riboja skaičiavimas:

| variantas | operacijų | baitų iš atminties | intensyvumas | riboja |
|-----------|-----------|--------------------|--------------|--------|
| i-k-j (`openmp-opt` su 1 gija) | 64·10⁹ | ≈ 256 GB | **0,25 op/B** | atmintis (skaito 54 GB/s – 93 % pralaidumo) |
| blokai (`seq-opt`) | 64·10⁹ | ≈ 1,6 GB | **≈ 40 op/B** | skaičiavimas |

![Operacijų vienam atminties baitui (log. skalė): i-k-j žemiau kompiuterio balanso – laukia atminties; blokinis variantas daug aukščiau – jo greitis nuo atminties nebepriklauso](docs/img/slides/intensyvumas.png)

`seq-opt` vienai operacijai skiria pastovias 0,025 ns (≈ 40 mlrd. operacijų per sekundę vienu branduoliu). Šis skaičiavimo branduolys (`include/gemm.hpp`) vėliau naudojamas MPI optimizuotame variante, o OpenMP optimizuotas variantas remiasi ta pačia idėja (blokai podėlyje, 4 × 16 mikrobranduolys registruose), bet turi savo realizaciją. Todėl optimizuotų variantų palyginimas su baziniais rodo ir atminties, ir lygiagretumo efektą.

## 5. Analizės įrankiai

Tyrimo dalyse lygiagretumo efektyvumas vertinamas keliais dėsniais. Čia pateikiami jų apibrėžimai.

**Pagreitėjimas** S(N) = T(1) / T(N) – to paties varianto laikas su 1 gija (procesu), padalytas iš laiko su N. Lyginama su tuo pačiu variantu, o ne su `seq-base`, kitaip atminties efektas būtų painiojamas su lygiagretumu.

**Amdalo dėsnis.** Jei dalis f programos vykdoma tik nuosekliai, tai

    S(N) = 1 / (f + (1 − f) / N),     S(∞) = 1 / f

Mūsų uždavinyje nuosekli dalis žinoma iš anksto – B pildymas, todėl f ne derinama prie kreivės, o **išmatuojama**: f = fill2 / total vienos gijos paleidime. B pildymas auga kaip n², o daugyba – kaip n³, todėl f mažėja didėjant matricoms (**Gustafsono dėsnis**).

**Karpo–Flato metrika** e = (1/S − 1/N) / (1 − 1/N) – nuoseklioji dalis, kuri paaiškintų išmatuotą pagreitėjimą. Jei e ≈ f, viską paaiškina Amdalo dėsnis; jei e > f, yra kitų nuostolių (atmintis, sinchronizacija, dažnis).

**Nevienodi branduoliai.** Amdalo dėsnis laiko branduolius vienodais, bet M2 Pro taupusis branduolys šiame uždavinyje atlieka tik r ≈ 0,15 našiojo darbo (išmatuota: 1,60 s našiajame ir 10,85 s taupiajame branduolyje, n = 4000). N gijų verta

    N_eff = min(N, 8) + r · max(N − 8, 0)

našiųjų branduolių – 12 gijų atitinka tik ≈ 8,6.

**Kas toliau.** 1 dalyje šie įrankiai pritaikomi OpenMP variantams, 2 dalyje – MPI, 3 dalyje – vaizdo plokštei.
