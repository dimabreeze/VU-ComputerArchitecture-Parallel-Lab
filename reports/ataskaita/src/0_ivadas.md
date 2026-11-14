title: Lygiagretusis matricų daugybos programavimas
subtitle: OpenMP, MPI ir OpenCL kompiuteryje Apple M2 Pro
part: Bendroji dalis

# Įvadas

!!! fact "Svarbiausi faktai"
    - Visi matavimai atlikti **įprastai naudojamame kompiuteryje**: prieš paleidimus kitos programos jau užėmė 7–12 GB iš 16 GB operatyviosios atminties ir apkrovė 1,4–1,7 procesoriaus branduolio (2 skyrius). Rezultatai – pakartotinių paleidimų medianos.
    - Vienos gijos programos **macOS nelaiko viename branduolyje**: gija nuolat perkeliama tarp 8 našiųjų branduolių, nors iš viso užima lygiai vieną (3 skyrius).
    - Paprasta daugyba (i-j-z) **staiga sulėtėja trigubai, kai n > 3547**: kiekvienas stulpelio žingsnis atneša naują podėlio eilutę (panaudojama 3 %) ir beveik visada naują puslapį, kurių TLB (≈ 3072 įrašai, išmatuota) nebesutalpina (4 skyrius).
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
| **Interaktyvi versija** ([dimabreeze.github.io/VU-ComputerArchitecture-Parallel-Lab](https://dimabreeze.github.io/VU-ComputerArchitecture-Parallel-Lab/)) | lenktynės, branduolių apkrovos grotuvas, visi matavimai, Amdalo dėsnis ir TLB riba – su tais pačiais duomenimis |

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

- **Fazės matuojamos atskirai** (A pildymas `fill1`, B pildymas `fill2`, daugyba `mul`, `total`) monotoniniu laikrodžiu; kiekviena konfigūracija – 3–7 paleidimai, naudojama mediana.
- **Visi variantai generuoja tuos pačius duomenis** (deterministinė elemento funkcija nuo sėklos ir indeksų), todėl laikus galima tiesiogiai lyginti, o rezultatą – patikrinti (`--verify`).
- Matricos kvadratinės: n = k = p, nuo 2000 iki 16000. Matavimų serija – `scripts/bench.sh`, duomenys – `results/csv/`.

## 3. Kaip stebime branduolių apkrovą

Programos `btop` ir `macmon` rodo kiekvieno branduolio ir vaizdo plokštės apkrovą; jų vaizdas užfiksuojamas daugybos viduryje (`make monitor`), procesų sąraše paliekama tik matuojama programa. `btop` numeracija: **C0–C3 – taupieji (E), C4–C11 – našieji (P) branduoliai.** Šilumos žemėlapiuose kas 100 ms įrašyta kiekvieno branduolio (GPU, P7…P0, E3…E0) apkrova ir dažnis, pažymėtos programos fazės, o antraštėje – kiek branduolių programa užėmė vidutiniškai (procesoriaus laikas / trukmė).

![seq-base (n = 3000): viena gija nuolat perkeliama tarp našiųjų branduolių, iš viso – lygiai 1,0 branduolio](docs/img/load/seq-base.png)
![seq-opt (n = 8000): vaizdas toks pat – 1,0 branduolio, nors skaičiuoja 125 kartus greičiau](docs/img/load/seq-opt.png)

Du dalykai, svarbūs skaitant tolesnių dalių vaizdus:

- **vienos gijos programa nėra „prikalta“ prie branduolio** – macOS ją perkelia tarp P0–P7, todėl vaizde ji atrodo kaip kelių branduolių dalinė apkrova;
- **apkrova nerodo naudingo darbo** – `seq-base` ir `seq-opt` abu užima lygiai vieną branduolį, bet `seq-base` didžiąją laiko dalį laukia duomenų iš atminties, o laukimas apkrovoje matomas kaip „darbas“.

## 4. Bendras atskaitos taškas: nuosekli daugyba ir atmintis

Visų dalių pradžia – ta pati nuosekli programa. Ji parodo, kad greitį pirmiausia lemia ne skaičiavimas, o tai, iš kurio atminties lygio ateina duomenys. Išsamiau: `docs/tlb_cliff.md`, `docs/memory.md`, `docs/seq_opt.md`.

### 4.1 seq-base ir staigus sulėtėjimas ties n ≈ 3547

`seq-base` daugina tvarka i-j-z: A eilutė dauginama iš B **stulpelio**. Matricos saugomos eilutėmis, todėl gretimi stulpelio elementai nutolę 16 000 B (n = 4000). Kiekvienas žingsnis **visada** atneša naują 128 B podėlio eilutę, iš kurios panaudojama **tik 3 %**, ir **97,7 % atvejų** patenka į naują 16 KB puslapį.

![Matricos atmintyje (n = 4000): B eilutė (16 000 B) trumpesnė už puslapį (16 384 B), todėl eilučių ir puslapių ribos „slenka“](docs/img/slides/atmintis-isdestymas.png)

![Vienas B stulpelio žingsnis kerta podėlio eilutės ribą visada, puslapio ribą – 97,7 % atvejų](docs/img/slides/ribu-kirtimas.png)

Kol visi stulpelio puslapiai telpa TLB (adresų vertimo buferyje), vertimai randami greitai. Stulpelis paliečia ≈ 4·n² / 16384 puslapių, o L2 TLB talpa – ≈ 3072 įrašai:

    4 · n² / 16384 > 3072   ⇒   n > 3547

![Pastebėta: seq-base, openmp-base ir mpi-base (1 gija) laikas pagal operacijų skaičių – ties n ≈ 3547 tiesė lūžta, viena operacija pabrangsta nuo 1,1 iki 3,2 ns](reports/ataskaita/img/tlb/plots/size-linear.png)
![Išmatuota: kai paliečiama daugiau nei 3072 puslapių, vieno skaitymo laikas šokteli nuo 7,7 iki 16,3 ns – tai L2 TLB riba](docs/img/memory/tlb.png)

Visi trys baziniai variantai guli ant tos pačios kreivės: tai atminties, ne lygiagretumo savybė.

### 4.2 Išmatuota atminties hierarchija

| lygis | dydis | atsitiktinio skaitymo delsa |
|-------|-------|-----------------------------|
| L1 podėlis | 128 KB | **0,88 ns** (≈ 3 taktai) |
| L2 podėlis | 16 MB | **5,6 ns** (≈ 20 taktų) |
| operatyvioji atmintis | 16 GB | **≈ 117 ns** (≈ 400 taktų, ≈ 130 kartų lėčiau už L1) |
| TLB praleidimas | > 3072 puslapiai | **+8,6 ns** |

Nuoseklaus skaitymo pralaidumas – **58 GB/s viena gija** ir ne daugiau kaip **≈ 122 GB/s** visomis gijomis. Matuota atskira programa `memlat` (`make memory`).

### 4.3 seq-opt: tas pats skaičiavimas, kitas darbas su atmintimi

`seq-opt` atlieka tas pačias operacijas, bet **skaido matricas blokais** (B blokas 1 MB lieka L2), **pakuoja** B bloką į ištisinį buferį (skaitoma tik nuosekliai, TLB problema išnyksta) ir **kaupia rezultatą registruose** (4 × 16 C fragmentas). Šis skaičiavimo branduolys (`include/gemm.hpp`) naudojamas ir MPI optimizuotame variante; OpenMP optimizuotas variantas remiasi ta pačia idėja, bet turi savo realizaciją.

![seq-base ir seq-opt laikas pagal matricos dydį (abi ašys logaritminės): seq-opt tiesė lygiagreti n³ – vienos operacijos kaina pastovi nuo n = 2000 iki 16000](reports/ataskaita/img/seqopt/plots/size.png){: .narrow }

**Kodėl 125 kartus (n = 4000: 208 s → 1,66 s).** Blokavimas keičia ne operacijų skaičių, o tai, kiek baitų atnešama iš atminties. Patogu tai išreikšti **aritmetiniu intensyvumu** – operacijomis vienam atneštam baitui. Kompiuterio balansas ≈ 0,7 op/B (≈ 40·10⁹ op/s vienam branduoliui ir 58 GB/s): mažiau – algoritmas laukia atminties, daugiau – riboja skaičiavimas.

| tvarka | baitų iš atminties (n = 4000, įvertis) | intensyvumas | riboja |
|--------|-----------------------------------------|--------------|--------|
| i-k-j (kiekvienai A eilutei – visa B) | ≈ 256 GB | **0,25 op/B** | atmintis (viena gija skaito 54 GB/s – 93 % pralaidumo) |
| blokai (`seq-opt`) | ≈ 1,6 GB (≈ 160 kartų mažiau) | **≈ 40 op/B** | skaičiavimas (0,025 ns operacijai) |

## 5. Analizės įrankiai

**Pagreitėjimas** S(N) = T(1) / T(N) – lyginamas to paties varianto laikas su 1 ir N gijų (procesų), ne su `seq-base`, kad atminties efektas nebūtų painiojamas su lygiagretumu.

**Amdalo dėsnis.** Jei dalis f vykdoma tik nuosekliai, S(N) = 1 / (f + (1 − f) / N), o riba S(∞) = 1 / f. Mūsų uždavinyje nuoseklioji dalis – B pildymas, todėl f **išmatuojama**: f = fill2 / total vienos gijos paleidime. Ji mažėja didėjant matricoms, nes pildymas auga kaip n², o daugyba – kaip n³ (**Gustafsono dėsnis**). Jei pagreitėjimas mažesnis, nei leidžia dėsnis, likusius nuostolius įvertina **Karpo–Flato metrika** – iš pagreitėjimo apskaičiuota nuoseklioji dalis e. Taupusis branduolys šiame uždavinyje atlieka tik ≈ 0,15 našiojo darbo (išmatuota), todėl 12 gijų verta tik ≈ 8,6 našiojo branduolio. Išsamiau: `docs/amdahl.md`.
