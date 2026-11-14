title: OpenCL: skaičiavimai vaizdo plokštėje
subtitle: naive → coalesced → tiled → regblock
part: 3 dalis

# 3 dalis. OpenCL

!!! fact "Svarbiausi faktai"
    - Ta pati vaizdo plokštė, tik kitaip parašytas kernelis: daugyba **27 kartus greitesnė** (n = 4000: 3,87 s → 0,143 s), o didžiausią žingsnį (**7×**) davė vien **kreipinių į atmintį tvarka** (2 skyrius).
    - **Tylioji klaida:** vienas `naive` paleidimas (n = 14000) „baigėsi“ per **0,22 s** – greičiau, nei vaizdo plokštė fiziškai gali. Kernelis buvo nutrauktas, o netikras laikas pateko į rezultatus; dabar kiekvienas paleidimas tikrinamas (3 skyrius).
    - Daugybos metu **vaizdo plokštė apkrauta 100 %**, o programa vidutiniškai užima tik **0,11 procesoriaus branduolio** (4 skyrius).
    - n = 4000 vaizdo plokštė daugina greičiau nei procesoriaus variantai (**0,143 s** prieš 0,218 s `openmp-opt2`), bet nuoseklus B pildymas jau sudaro **27 %** viso laiko (6 skyrius).

## 1. OpenCL modelis ir užduoties įgyvendinimas

OpenCL programą sudaro **pagrindinė programa** (host – C++ programa procesoriuje) ir **kerneliai** – OpenCL C funkcijos, kurias pagrindinė programa vykdymo metu sukompiliuoja ir paleidžia **įrenginyje** (device – M2 Pro vaizdo plokštėje). Kernelis paleidžiamas indeksų erdvei **NDRange**: kiekvienas jos taškas – atskiras darbo elementas, o darbo elementai grupuojami į darbo grupes. Atminties lygiai atitinka procesoriaus hierarchiją (Bendrosios dalies 4 skyrius), todėl visas optimizavimas – tai siekis imti duomenis iš kuo artimesnio lygio.

| sąvoka | reikšmė šiame darbe |
|--------|---------------------|
| **darbo elementas** (work-item) | vienas kernelio vykdymas su savo indeksu `get_global_id(d)` |
| **darbo grupė** (work-group) | čia 16 × 16 darbo elementų viename skaičiavimo vienete; dalijasi lokaliąja atmintimi, sinchronizuojasi `barrier` |
| **skaičiavimo vienetas** (compute unit) | vaizdo plokštės „branduolys“; M2 Pro jų turi **19** |
| **globalioji / lokalioji / privati atmintis** | bendra 16 GB operatyvioji atmintis / maža luste esanti grupės atmintis / registrai |

**Užduoties reikalavimai.** A pildoma **lygiagrečiai** vaizdo plokštėje kerneliu `fill_matrix1` (`src/opencl/common.cl`; vienas darbo elementas – vienas elementas), B – **nuosekliai** procesoriuje ta pačia funkcija kaip procesoriaus variantuose, C = A × B – **lygiagrečiai** vaizdo plokštėje kerneliu `matmul`. Generatorius tas pats deterministinis, todėl rezultatus galima patikrinti (`--verify`).

**Pagrindinės programos žingsniai** (`src/opencl/base.cpp`, `src/opencl/opt.cpp`):

1. platforma ir vaizdo plokštė – `clGetPlatformIDs`, `clGetDeviceIDs(CL_DEVICE_TYPE_GPU)`;
2. kontekstas ir komandų eilė – `clCreateContext`, `clCreateCommandQueue`;
3. kernelių kompiliavimas iš `.cl` failo – `clCreateProgramWithSource`, `clBuildProgram`, `clCreateKernel`;
4. buferiai matricoms – `clCreateBuffer`;
5. paleidimas – `clSetKernelArg`, `clEnqueueNDRangeKernel`;
6. rezultato nuskaitymas ir laukimas – `clEnqueueReadBuffer` / `clEnqueueMapBuffer`, `clFinish`.

Naudojama Apple OpenCL 1.2 (pažymėta kaip pasenusi, bet veikia). Fazės matuojamos kaip aprašyta Bendrosios dalies 2.3 skyriuje; `mul` apima ir sinchronizaciją.

## 2. Kerneliai: naive → coalesced → tiled → regblock

Visi trys optimizuoti kerneliai naudoja tą pačią pagrindinę programą (`opencl-opt`), todėl laiko skirtumą lemia tik kernelis. Operacija – viena daugybos ir sudėties pora, pralaidumas = n · k · p / mul.

| kernelis | kas pakeista | mul, n = 4000 | žingsnio nauda | 10⁹ op/s |
|----------|--------------|---------------|----------------|----------|
| `naive` | vienas darbo elementas – vienas C elementas | 3,87 s | – | 16,6 |
| `coalesced` | stulpelis 0 dimensijoje, `mad24` | 0,545 s | **7,1×** | 117 |
| `tiled` | 16 × 16 blokai lokaliojoje atmintyje | 0,375 s | **1,45×** | 171 |
| `regblock` | 8 × 8 C blokas darbo elementui registruose | 0,143 s | **2,6×** | 446 |

**naive** – tiesioginis i-j-z vertimas, du išorinius ciklus pakeičia NDRange:

```c
const int i = get_global_id(0);   // eilutė – 0 dimensijoje
const int j = get_global_id(1);
if (i >= n || j >= p) return;
int sum = 0;
for (int z = 0; z < k; ++z) sum += m1[i * k + z] * m2[z * p + j];
m3[i * p + j] = sum;
```

Jis pasiekia tik 16·10⁹ operacijų per sekundę – 18 kartų lėčiau nei `openmp-opt2` su 8 gijomis.

**coalesced (7×).** Vaizdo plokštė kartu vykdo darbo elementus, kurie skiriasi **0 dimensijos** indeksu; jei jie kreipiasi į gretimus adresus, kreipiniai sujungiami į vieną transakciją (**sujungti kreipiniai**). `naive` 0 dimensijoje turi eilutę, todėl kaimynai skaito A žingsniu k – kiekvienas savo podėlio eilutę. Pakanka sukeisti dimensijas:

```c
const int j = get_global_id(0);            // stulpelis – 0 dimensijoje
const int i = row0 + get_global_id(1);
for (int z = 0; z < k; ++z) sum = MAD(a[z], b[(size_t)z * p], sum);
```

Dabar kaimynai viena transakcija skaito gretimus B elementus, o tą patį A elementą skaito visi kartu. Kartu įvesta `mad24()` – greita 24 bitų sveikųjų skaičių daugybos ir sudėties komanda (įjungiama tik kai x < 2²³, rezultatas nesikeičia).

**tiled (1,45×).** 16 × 16 darbo grupė bendrai įkelia po A ir B **bloką** į lokaliąją atmintį, sinchronizuojasi `barrier` ir kiekvieną elementą panaudoja 16 kartų. Globaliosios atminties srautas krenta 16 kartų, bet laikas – tik 1,45 karto: tikėtina, kad dalį pakartotinių skaitymų jau aptarnavo podėliai, o kiekvienai operacijai vis dar reikia dviejų lokaliosios atminties skaitymų.

**regblock (2,6×).** Kiekvienas darbo elementas skaičiuoja **8 × 8 C elementus** registruose (grupė – 128 × 128 bloką). Per z žingsnį jis nuskaito 8 + 8 reikšmes ir atlieka 64 operacijas – 8 kartus mažiau skaitymų vienai operacijai. Tai `seq-opt` mikrobranduolio (Bendrosios dalies 4.3 skyrius) atitikmuo:

```c
for (int z = 0; z < TK; ++z) {
    int a[WM], b[WN];                                        // registrai
    for (int r = 0; r < WM; ++r) a[r] = As[ly + LY * r][z];  // 8 A reikšmės
    for (int c = 0; c < WN; ++c) b[c] = Bs[z][lx + LX * c];  // 8 B reikšmės
    for (int r = 0; r < WM; ++r)
        for (int c = 0; c < WN; ++c) acc[r][c] = MAD(a[r], b[c], acc[r][c]);
}
```

Bloko dydžiai yra makrokomandos; išbandžius 8 konfigūracijas n = 4000, pasirinkta 8 × 8 elementų darbo elementui, 16 × 16 darbo grupė ir TK = 16. Išsamiau: `docs/opencl_opt.md`.

## 3. Pagrindinė programa ir tylioji klaida

`opt.cpp` nuo `base.cpp` skiriasi keturiais dalykais:

- **buferiai be kopijavimo** – procesorius ir vaizdo plokštė turi bendrą atmintį, todėl buferiai kuriami ant pagrindinės programos matricų (`CL_MEM_USE_HOST_PTR` + map/unmap), o ne kopijuojami (n = 4000 `base.cpp` kopijuoja po 64 MB); paruošimas ir sinchronizacija trunka ≈ 3 ms;
- **paleidimas eilučių juostomis** – po ≈ 2³⁴ operacijų (bent viena 128 eilučių blokų eilutė), todėl kiekviena komanda trunka dešimtis milisekundžių, o ne minutes (n = 14000: 110 paleidimų po ≈ 0,05 s);
- **kiekvienas paleidimas tikrinamas** – kiekvieno kernelio įvykio būsena (`CL_EVENT_COMMAND_EXECUTION_STATUS`) ir **atrankinis patikrinimas**: procesorius perskaičiuoja 4 kampus ir 256 atsitiktinius C elementus. Nepavykęs paleidimas žymimas `FAILED` ir nepatenka į CSV;
- **profiliavimo pataisa** – Apple OpenCL laiko žymas grąžina ne nanosekundėmis, kaip reikalauja standartas, o `mach_absolute_time` taktais (1 taktas = 125/3 ≈ 41,67 ns), todėl kernelio laikas išeidavo 41,7 karto per mažas; dabar perskaičiuojama pagal `mach_timebase_info`.

!!! fact "Tylioji klaida: 168 s darbas „atliktas“ per 0,22 s"
    Matuojant `naive` su n = 14000, du iš trijų paleidimų truko po 168 s, o 2-asis „truko“ **0,22 s**:

        2,7·10¹² / 0,22 s ≈ 12·10¹² operacijų/s  >  ≈ 6,8·10¹² (vaizdo plokštės FP32 riba)

    Kernelis šio darbo atlikti negalėjo – jis buvo nutrauktas. Tačiau `base.cpp` tada netikrino kernelio būsenos, o blokuojantis `clEnqueueReadBuffer` grąžina sėkmę net nepavykus kernelio komandai, todėl **netikras laikas pateko į rezultatus**. Greičiausiai priežastis – macOS apsauga nuo ilgų komandų vaizdo plokštėje, kuri piešia ekraną (kernelis veikė beveik 3 minutes); sistemos žurnale to patvirtinti nepavyko. Dabar toks paleidimas žymimas `FAILED`, o `opencl-opt` ilgų komandų išvengia juostomis.

Išsamiau: `docs/opencl_opt.md`.

## 4. Apkrova vykdymo metu

Apkrova stebėta Bendrosios dalies 3 skyriuje aprašytu būdu, paleidus regblock su n = 16000 (daugyba 9,63 s, visa programa 11,2 s).

![btop daugybos metu: procesoriaus branduoliai beveik laisvi, opencl-opt naudoja 0,2 % procesoriaus, GPU juosta – 100 %](docs/img/screens/opencl-regblock-btop.png)
![macmon tuo pačiu metu: GPU apkrauta 100 % ties 1398 MHz ir naudoja ≈ 15,7 W](docs/img/screens/opencl-regblock-macmon.png)

![opencl-regblock šilumos žemėlapis: daugybos metu ištisai apkrauta tik GPU eilutė; ryškus P0 prieš daugybą – nuoseklus B pildymas, trumpi P branduolių blyksniai ≈ 7–9 s – ne šios programos](docs/img/load/opencl-regblock.png){: .wide }

- **Vaizdo plokštė apkrauta 100 %.** Kas 100 ms įrašytų reikšmių medianos daugybos metu: aktyvumas 100 %, dažnis **1398 MHz** (prieš paleidimą – 444 MHz), galia **≈ 15,4 W** (prieš paleidimą – ≈ 0,1 W).
- **Procesorius beveik nedirba.** Pagal `/usr/bin/time` programa sunaudojo 0,98 s naudotojo ir 0,30 s sistemos laiko per 11,19 s – vidutiniškai **0,11 branduolio**. Tas naudotojo laikas beveik lygus B pildymo trukmei (0,90 s): žemėlapyje tai vienas ryškus P branduolys prieš daugybą. Daugybos metu procesorius tik pateikia juostas ir laukia.
- **Kitas aktyvumas – ne šios programos:** E branduolius apkrauna fono programos (Bendrosios dalies 2 skyrius); trumpi P branduolių blyksniai ≈ 7–9 s greičiausiai susiję su ekrano vaizdo fiksavimu.
- **Atmintis** išauga ≈ 3 GB – tiek užima trys 16000 × 16000 matricos; antros kopijos vaizdo plokštei nėra.

## 5. Laiko priklausomybė nuo dydžio ir „branduolių skaičiaus“

![Visas programos laikas pagal matricos dydį (abi ašys logaritminės): visos keturios tiesės lygiagrečios n³ tiesei, atstumas tarp naive ir regblock beveik pastovus](reports/ataskaita/img/ocl/plots/size.png)

| n | naive | coalesced | tiled | regblock | regblock, palyginti su naive |
|---|-------|-----------|-------|----------|------------------------------|
| 2000 | 0,483 s | 0,070 s | 0,048 s | 0,020 s | 24× |
| 4000 | 3,87 s | 0,545 s | 0,375 s | 0,143 s | 27× |
| 8000 | 31,0 s | 4,39 s | 3,01 s | 1,19 s | 26× |
| 14000 | 168 s | 23,7 s | 16,1 s | 5,77 s | 29× |
| 16000 | – | 35,4 s | 24,0 s | 9,92 s | |

Lentelėje – `mul` medianos. Laikas auga kaip n³, **be lūžių**: priešingai nei procesoriuje ties n ≈ 3547 (Bendrosios dalies 4.1 skyrius), vienos operacijos kaina pastovi. `naive`, `coalesced` ir `tiled` pralaidumas visame intervale kinta ne daugiau kaip ±2 %, regblock – tarp 376 ir 475·10⁹ op/s (tikėtina, dėl fono apkrovos ir to, kad ta pati vaizdo plokštė piešia ekraną). **Pagreitėjimas išlieka 24–29× visuose dydžiuose**, vadinasi, `naive` problema buvo ne dydis, o kreipinių tvarka. `naive` su n = 16000 nematuotas – vienas kernelis truktų ≈ 4 minutes.

**„Branduolių skaičius“.** Užduoties reikalavimas (a) vaizdo plokštei tiesiogiai netaikomas: ji turi fiksuotus **19 skaičiavimo vienetų**, kuriems darbo grupes paskirsto pats OpenCL. Lygiagretumą lemia NDRange ir darbo grupės dydis. regblock n = 4000 paleidžia ≈ 60 kartų mažiau darbo elementų nei `naive` (4 juostos po 65 536 vietoj 16·10⁶), bet yra 27 kartus greitesnis: svarbu ne darbo elementų kiekis, o kiek kiekvienas jų skaičiuoja su duomenimis registruose. Per maža grupė kenkia – su 8 × 8 darbo grupe vietoj 16 × 16 daugyba lėtesnė 40 % (0,201 s), tikėtina, nes per mažai darbo elementų atminties delsai paslėpti.

Išsamiau: `reports/06-opencl/README.md`.

## 6. Vaizdo plokštė ir procesorius

| variantas (n = 4000) | mul | total |
|----------------------|-----|-------|
| `seq-opt` (1 branduolys) | 1,58 s | 1,66 s |
| `openmp-opt2` (8 gijos) | 0,218 s | 0,276 s |
| `mpi-opt` (8 procesai) | 0,242 s | 0,300 s |
| **`opencl-opt` regblock** | **0,143 s** | **0,220 s** |

Daugybą vaizdo plokštė atlieka 1,5 karto greičiau nei `openmp-opt2` ir 11 kartų greičiau nei vienas branduolys, bet visa programa – tik 1,25 karto, nes kitos fazės naudos negauna:

- **nuoseklus B pildymas** (0,058 s) sudaro **27 %** viso laiko; programa negali būti trumpesnė už A ir B pildymą (0,076 s). Didėjant n jo dalis mažėja (Bendrosios dalies 5 skyrius): n = 16000 – tik 8 %, ir ten atotrūkis didžiausias – 11,1 s, palyginti su 16,3 s geriausio `openmp-opt2` (12 gijų);
- **A pildymas vaizdo plokštėje** (0,017 s) lėtesnis nei 8 gijomis (0,002 s) – tikėtina, kad vyrauja paleidimo ir sinchronizacijos sąnaudos;
- **int32 aritmetika** vaizdo plokštėje lėtesnė nei FP32; kernelis su `float` greičiausiai būtų greitesnis, bet tai pakeistų užduoties duomenų tipą.

## Išvados

1. **Užduotis įgyvendinta vaizdo plokštėje:** A pildoma lygiagrečiai kerneliu, B – nuosekliai procesoriuje, C = A × B – lygiagrečiai kerneliu; kiekvienas paleidimas patikrinamas.
2. **Vaizdo plokštės greitį lemia kreipinių į atmintį tvarka:** keičiant tik kernelį, pralaidumas išauga nuo 16·10⁹ iki ≈ 450·10⁹ operacijų per sekundę (27×); vien teisinga NDRange dimensija davė 7×.
3. **Laikas auga kaip n³ be lūžių,** pagreitėjimas visuose dydžiuose 24–29×. Skaičiavimo vienetų skaičius fiksuotas (19), lygiagretumą lemia NDRange ir darbo grupės dydis.
4. **Apkrova:** daugybos metu vaizdo plokštė apkrauta 100 % (1398 MHz, ≈ 15 W), procesorius – vidutiniškai 0,11 branduolio, beveik vien B pildymui.
5. **Vaizdo plokštės klaidos gali būti tylios:** be būsenos ir rezultato patikrinimų nutrauktas kernelis atrodė 760 kartų greitesnis. Vaizdo plokštė – greičiausias variantas, bet nuoseklus B pildymas (27 % n = 4000) riboja tolesnę naudą.
