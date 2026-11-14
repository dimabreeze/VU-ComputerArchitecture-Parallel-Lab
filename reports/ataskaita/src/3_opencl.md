title: OpenCL: skaičiavimai vaizdo plokštėje
subtitle: naive → coalesced → tiled → regblock
part: 3 dalis

# 3 dalis. OpenCL

!!! fact "Svarbiausi faktai"
    - Ta pati vaizdo plokštė, tik kitaip parašytas kernelis: daugyba **27 kartus greitesnė** (n = 4000: 3,87 s → 0,143 s), pralaidumas išauga nuo **16·10⁹ iki ≈ 450·10⁹** operacijų per sekundę (2 skyrius).
    - Didžiausią vieno žingsnio naudą (**7×**) davė ne skaičiavimo pakeitimas, o **kreipinių į atmintį tvarka**: kai stulpelio indeksas yra 0 NDRange dimensijoje, gretimi darbo elementai skaito gretimus adresus viena transakcija (2 skyrius).
    - **Tylioji klaida:** vienas `naive` paleidimas (n = 14000) „baigėsi“ per **0,22 s** – tai būtų 12·10¹² operacijų per sekundę, daugiau, nei vaizdo plokštė gali atlikti. Kernelis buvo nutrauktas, o netikras laikas pateko į rezultatus. Dabar kiekvienas paleidimas tikrinamas, nepavykę žymimi `FAILED` (3 skyrius).
    - Daugybos metu **vaizdo plokštė apkrauta 100 %** (1398 MHz, ≈ 15 W), o programa vidutiniškai užima tik **0,11 procesoriaus branduolio**: procesorius dirba beveik tik nuosekliai pildydamas B (4 skyrius).
    - n = 4000 vaizdo plokštė daugina per **0,143 s** – greičiau nei procesoriaus variantai (`openmp-opt2` su 8 gijomis – 0,218 s), tačiau nuoseklus B pildymas dabar sudaro jau **27 %** viso laiko (6 skyrius).

## 1. OpenCL modelis ir užduoties įgyvendinimas

### 1.1 Pagrindinės sąvokos

OpenCL programą sudaro dvi dalys. **Pagrindinė programa** (host, čia – C++ programa procesoriuje) paruošia duomenis ir siunčia komandas **įrenginiui** (device, čia – M2 Pro vaizdo plokštei). Įrenginyje vykdomos funkcijos vadinamos **kerneliais**; jie rašomi OpenCL C kalba atskiruose `.cl` failuose ir kompiliuojami programos vykdymo metu.

| sąvoka | reikšmė šiame darbe |
|--------|---------------------|
| **NDRange** | indeksų erdvė, kuriai paleidžiamas kernelis; čia dvimatė – C eilutės ir stulpeliai |
| **darbo elementas** (work-item) | vienas kernelio vykdymas su savo indeksu `get_global_id(d)` – tarsi labai lengva gija |
| **darbo grupė** (work-group) | darbo elementai (čia 16 × 16 = 256), vykdomi viename skaičiavimo vienete; gali dalytis lokaliąja atmintimi ir sinchronizuotis (`barrier`) |
| **skaičiavimo vienetas** (compute unit) | vaizdo plokštės „branduolys“; M2 Pro jų turi **19** |
| **globalioji atmintis** (`__global`) | didelė, bet lėta; čia tai ta pati 16 GB operatyvioji atmintis, kurią naudoja procesorius |
| **lokalioji atmintis** (`__local`) | maža luste esanti atmintis, bendra vienai darbo grupei |
| **privati atmintis** | kiekvieno darbo elemento kintamieji **registruose** – greičiausias lygis |

Šie lygiai atitinka procesoriaus atminties hierarchiją (Bendrosios dalies 4 skyrius): visas tolesnis optimizavimas – tai siekis imti duomenis iš kuo artimesnio lygio.

### 1.2 Kaip įvykdyti užduoties reikalavimai

| užduoties žingsnis | kur vykdoma | kaip |
|--------------------|-------------|------|
| A (n × k) pildymas skaičiais [1, x] – **lygiagrečiai** | vaizdo plokštėje | kernelis `fill_matrix1`: vienas darbo elementas – vienas A elementas |
| B (k × p) pildymas ±1 – **nuosekliai** | procesoriuje, viena gija | ta pati `fill_matrix2_sequential` kaip procesoriaus variantuose |
| C = A × B – **lygiagrečiai** | vaizdo plokštėje | kernelis `matmul` (keturi variantai, 2 skyrius) |

Atsitiktiniai skaičiai generuojami tuo pačiu deterministiniu generatoriumi (`splitmix64`) kaip procesoriaus variantuose, todėl A, B ir C sutampa su jų rezultatais ir gali būti patikrinti (`--verify`). Optimizuotų variantų A pildymo kernelis (`src/opencl/common.cl`):

```c
__kernel void fill_matrix1(__global int* m1, int n, int k, int x, ulong seed) {
    const int j = get_global_id(0);   // 0 dimensija – stulpelis
    const int i = get_global_id(1);   // 1 dimensija – eilutė
    if (i >= n || j >= k) return;
    const ulong idx = (ulong)i * (ulong)k + (ulong)j;
    m1[idx] = 1 + (int)(splitmix64(seed ^ (idx * 0xD1B54A32D192ED03UL)) % (ulong)x);
}
```

Elementas priklauso tik nuo sėklos ir savo indekso, todėl darbo elementams nereikia jokio tarpusavio derinimo. Pradinis variantas (`src/opencl/base/naive.cl`) turi tokį pat kernelį, tik su eilute 0 dimensijoje; kodėl tai svarbu, paaiškinta 2.2 skyriuje.

### 1.3 Pagrindinės programos žingsniai

Pagrindinė programa (`src/opencl/base.cpp`, patobulinta – `src/opencl/opt.cpp`):

1. randa platformą ir vaizdo plokštę: `clGetPlatformIDs`, `clGetDeviceIDs(..., CL_DEVICE_TYPE_GPU, ...)`;
2. sukuria kontekstą ir komandų eilę: `clCreateContext`, `clCreateCommandQueue`;
3. nuskaito `.cl` failą ir jį sukompiliuoja: `clCreateProgramWithSource`, `clBuildProgram`, `clCreateKernel`;
4. sukuria buferius matricoms: `clCreateBuffer`;
5. nustato argumentus ir paleidžia kernelį: `clSetKernelArg`, `clEnqueueNDRangeKernel`;
6. nuskaito rezultatą ir laukia, kol eilė baigsis.

Pradinio varianto daugybos fazė:

```cpp
CL_CHECK(clEnqueueWriteBuffer(q, d2, CL_FALSE, 0, bytes2, m2.data.data(),
                              0, nullptr, nullptr));
// ... clSetKernelArg: d1, d2, d3, n, k, p
const size_t g_mul[2] = {static_cast<size_t>(a.n), static_cast<size_t>(a.p)};
CL_CHECK(clEnqueueNDRangeKernel(q, k_mul, 2, nullptr, g_mul, nullptr,
                                0, nullptr, &e_mul));
CL_CHECK(clEnqueueReadBuffer(q, d3, CL_TRUE, 0, bytes3, m3.data.data(),
                             0, nullptr, nullptr));
```

NDRange yra n × p – po vieną darbo elementą kiekvienam C elementui, darbo grupės dydį (`nullptr`) parenka tvarkyklė. Fazės matuojamos kaip aprašyta Bendrosios dalies 2.3 skyriuje: `mul` apima ir duomenų perdavimą, ir sinchronizaciją. Naudojama Apple OpenCL 1.2 – pažymėta kaip pasenusi (deprecated), bet veikia.

## 2. Kernelių optimizavimas: naive → coalesced → tiled → regblock

Kiekvienas žingsnis – atskiras kernelio failas, pagrindinė programa ta pati (`opencl-opt`), todėl laiko skirtumą lemia tik kernelis.

| kernelis | kas pakeista | mul, n = 4000 | žingsnio nauda | 10⁹ op/s |
|----------|--------------|---------------|----------------|-------------|
| `naive` | vienas darbo elementas – vienas C elementas | 3,87 s | – | 16,6 |
| `coalesced` | stulpelis 0 dimensijoje, `mad24`, grupė 16 × 16 | 0,545 s | **7,1×** | 117 |
| `tiled` | 16 × 16 blokai lokaliojoje atmintyje | 0,375 s | **1,45×** | 171 |
| `regblock` | 8 × 8 C blokas darbo elementui registruose | 0,143 s | **2,6×** | 446 |

„Operacija“ čia ir toliau – viena daugybos ir sudėties pora, pralaidumas (paskutinis stulpelis) skaičiuojamas taip:

    pralaidumas = n · k · p / mul        (n = 4000: 6,4·10¹⁰ operacijų)

### 2.1 naive: vienas darbo elementas – vienas C elementas

```c
const int i = get_global_id(0);   // eilutė – 0 dimensijoje
const int j = get_global_id(1);
if (i >= n || j >= p) return;
int sum = 0;
for (int z = 0; z < k; ++z) sum += m1[i * k + z] * m2[z * p + j];
m3[i * p + j] = sum;
```

Tai tiesioginis i-j-z algoritmo vertimas: du išorinius ciklus pakeičia NDRange. Kernelis pasiekia tik ≈ 16·10⁹ operacijų per sekundę – mažą dalį to, ką gali ši vaizdo plokštė (apie 6,8·10¹² FP32 operacijų per sekundę), ir **18 kartų lėčiau** nei `openmp-opt2` su 8 gijomis.

### 2.2 coalesced: tinkama NDRange dimensija (7×)

Vaizdo plokštė darbo elementus vykdo grupėmis, kurių nariai skiriasi **0 dimensijos** indeksu. Kai tokie kaimynai kreipiasi į gretimus adresus, kreipiniai sujungiami į vieną atminties transakciją (**sujungti kreipiniai**, coalesced). `naive` 0 dimensijoje turi eilutę i, todėl kaimynai skaito `m1[i*k + z]` žingsniu k (kiekvienas – savo podėlio eilutę) ir rašo `m3[i*p + j]` žingsniu p. `coalesced.cl` dimensijas sukeičia:

```c
const int j = get_global_id(0);            // stulpelis – 0 dimensijoje
const int i = row0 + get_global_id(1);
...
for (int z = 0; z < k; ++z) sum = MAD(a[z], b[(size_t)z * p], sum);
```

Dabar kaimynai viena transakcija skaito iš eilės einančius `B[z*p + j]`, o tą patį `A[i*k + z]` – visi kartu (vienas kreipinys, reikšmė transliuojama). Tai tas pats podėlio eilutės principas kaip procesoriuje (Bendrosios dalies 4.1 skyrius), tik eilutę čia vienu metu naudoja daug darbo elementų.

Kartu įvesta `mad24()` – sveikųjų skaičių daugybos ir sudėties komanda 24 bitų operandams, kurią vaizdo plokštė vykdo greitaisiais blokais. Pagrindinė programa ją įjungia (`-DUSE_MAD24`) tik kai x < 2²³: A reikšmės yra [1, x], B – ±1, todėl rezultatas lieka tas pats 32 bitų skaičius. Darbo grupė fiksuota – 16 × 16. Šis žingsnis sutrumpina daugybą **nuo 3,87 iki 0,545 s (7,1×)**.

### 2.3 tiled: blokai lokaliojoje atmintyje (1,45×)

`coalesced` kiekvieną A ir B elementą vis dar skaito iš globaliosios atminties k kartų. `tiled.cl` 16 × 16 darbo grupei paskiria 16 × 16 C **bloką**. Kiekviename 16 ilgio žingsnyje išilgai k grupė bendrai įkelia po vieną A ir B bloką į lokaliąją atmintį, sinchronizuojasi, ir kiekvienas įkeltas elementas panaudojamas 16 kartų:

```c
As[ly][lx] = (i < n && t + lx < k) ? A[(size_t)i * k + t + lx] : 0;   // už ribų – 0
Bs[ly][lx] = (t + ly < k && j < p) ? B[(size_t)(t + ly) * p + j] : 0;
barrier(CLK_LOCAL_MEM_FENCE);                                         // blokas įkeltas
for (int z = 0; z < TS; ++z) sum = MAD(As[ly][z], Bs[z][lx], sum);
barrier(CLK_LOCAL_MEM_FENCE);                                         // blokas panaudotas
```

Globaliosios atminties srautas sumažėja 16 kartų, bet laikas – tik 1,45 karto (0,545 → 0,375 s). Tikėtina, kad dalį pakartotinių `coalesced` skaitymų jau aptarnavo vaizdo plokštės podėliai, o `tiled` vis dar atlieka **du lokaliosios atminties skaitymus kiekvienai operacijai** ir du barjerus kiekviename žingsnyje. Ribojimas persikelia į lokaliąją atmintį.

### 2.4 regblock: blokavimas registruose (2,6×)

`regblock.cl` kiekvienam darbo elementui paskiria **8 × 8 C elementus**, laikomus registruose; 16 × 16 darbo grupė apskaičiuoja 128 × 128 C bloką. Kiekviename z žingsnyje darbo elementas iš lokaliosios atminties nuskaito 8 + 8 reikšmes ir atlieka su jomis 64 operacijas:

```c
for (int z = 0; z < TK; ++z) {
    int a[WM], b[WN];                                        // registrai
    for (int r = 0; r < WM; ++r) a[r] = As[ly + LY * r][z];  // 8 A reikšmės
    for (int c = 0; c < WN; ++c) b[c] = Bs[z][lx + LX * c];  // 8 B reikšmės
    for (int r = 0; r < WM; ++r)
        for (int c = 0; c < WN; ++c) acc[r][c] = MAD(a[r], b[c], acc[r][c]);
}
```

Skaitymų vienai operacijai sumažėja nuo 2 iki 16 / 64 = 0,25, t. y. 8 kartus – tai `seq-opt` mikrobranduolio (Bendrosios dalies 4.3 skyrius) atitikmuo vaizdo plokštėje. Darbo elementui (lx, ly) priklauso eilutės `ly + 16·r` ir stulpeliai `lx + 16·c`, todėl kaimynai skaito kaimyninius lokaliojo bloko stulpelius ir rašo į kaimyninius adresus; lokalusis A blokas papildytas vienu stulpeliu (`As[TM][TK + 1]`), kad nekiltų lokaliosios atminties bankų konfliktų.

Bloko dydžiai yra makrokomandos (`WM`, `WN` – blokas darbo elementui, `LX`, `LY` – darbo grupė, `TK` – bloko gylis), keičiamos neredaguojant failo, pvz. `CL_BUILD_OPTS="-DWM=8 -DWN=4"`. Derinimo rezultatai (`mul`, geriausias iš 3 paleidimų, ne mediana):

| WM × WN | darbo grupė | TK | n = 4000 | n = 8000 |
|---------|-------------|----|----------|----------|
| 4 × 4 | 16 × 16 | 16 | 0,157 s | 1,264 s |
| 4 × 8 | 16 × 16 | 16 | 0,153 s | |
| 8 × 4 | 16 × 16 | 16 | 0,145 s | 1,199 s |
| **8 × 8** | **16 × 16** | **16** | **0,143 s** (pasirinkta) | **1,184 s** |
| 8 × 8 | 8 × 8 | 16 | 0,201 s | |
| 8 × 8 | 16 × 16 | 8 | 0,145 s | |
| 4 × 4 | 16 × 16 | 32 | 0,171 s | |
| 8 × 4 | 32 × 8 | 16 | 0,149 s | |

Didesnis blokas registruose padeda vis mažiau (4 × 4 → 8 × 8: tik ≈ 10 %). Labiausiai rezultatą pablogina **per maža darbo grupė**: su 8 × 8 darbo elementų grupe daugyba lėtesnė 40 % (0,201 s) – žr. 5.2 skyrių.

## 3. Pagrindinės programos patobulinimai

Rezultatą keičia ne tik kernelis. `opt.cpp` nuo `base.cpp` skiriasi keturiais dalykais.

**Buferiai be kopijavimo.** M2 Pro procesorius ir vaizdo plokštė naudoja **tą pačią operatyviąją atmintį**, bet `base.cpp` vis tiek kopijuoja: A – atgal į pagrindinę programą, B – į įrenginį, C – atgal (n = 4000: po 64 MB). `opt.cpp` buferius kuria **be kopijavimo** (zero-copy), tiesiog ant pagrindinės programos matricų; `clEnqueueMapBuffer` / `Unmap` tik užtikrina, kad vaizdo plokštės įrašai būtų matomi procesoriui:

```cpp
cl_mem d1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, bytes1,
                           m1.data_ptr(), &err);
void* p1 = clEnqueueMapBuffer(q, d1, CL_TRUE, CL_MAP_READ, 0, bytes1,
                              0, nullptr, nullptr, &err);
CL_CHECK(clEnqueueUnmapMemObject(q, d1, p1, 0, nullptr, nullptr));
```

Buferių paruošimas ir sinchronizacija regblock variante užtrunka apie 3 ms.

**Paleidimas eilučių juostomis.** Daugyba skaidoma į kelis paleidimus – **eilučių juostas**. Juostoje – sveikas skaičius C blokų eilučių (regblock: po 128) ir apie 2³⁴ ≈ 1,7·10¹⁰ operacijų, bet ne mažiau kaip viena blokų eilutė:

```cpp
int band = static_cast<int>(std::min<unsigned long long>(a.n, BAND_OPS / per_row));
band = std::max(TM, band / TM * TM);       // sveikas blokų skaičius, bent vienas
for (int row0 = 0; row0 < a.n; row0 += band) {
    // ... globalusis dydis šiai juostai, argumentas row0
    CL_CHECK(clEnqueueNDRangeKernel(q, k_mul, 2, nullptr, gws, lws, 0, nullptr, &e));
    CL_CHECK(clFlush(q));                  // pateikti iškart
    e_mul.push_back(e);
}
```

n = 4000 tai 4 paleidimai po 1024 eilutes, n = 14000 – 110 paleidimų po 128 eilutes (≈ 2,5·10¹⁰ operacijų, ≈ 0,05 s kiekvienas). Kiekviena komanda trunka dešimtis milisekundžių, o ne minutes. Pagalbinis kernelis `geometry` kiekviename faile praneša darbo grupės ir C bloko dydžius, todėl viena pagrindinė programa aptarnauja visus kernelius.

**Kiekvieno paleidimo patikrinimas.** Kodėl jo reikia, parodė vienas matavimas:

!!! fact "Tylioji klaida: 168 s darbas „atliktas“ per 0,22 s"
    Matuojant `naive` su n = 14000 (3 pakartojimai), 1-asis ir 3-iasis paleidimai truko po 168 s, o 2-asis „truko“ **0,22 s**:

        2,7·10¹² / 0,22 s ≈ 12·10¹² operacijų/s  >  ≈ 6,8·10¹² (vaizdo plokštės FP32 riba)

    Kernelis šio darbo atlikti negalėjo – jis buvo nutrauktas arba atmestas. Tačiau `base.cpp` tada **netikrino kernelio būsenos**, o blokuojantis `clEnqueueReadBuffer` grąžina sėkmę net nepavykus kernelio komandai. Neteisingas C buvo „gautas“ su netikru laiku, ir šis laikas **pateko į rezultatus**. Toks laikas ≈ 760 kartų per geras.

    Greičiausiai priežastis – macOS apsauga nuo ilgų vaizdo plokštės komandų: vienas kernelis veikė beveik 3 minutes toje pačioje vaizdo plokštėje, kuri piešia ekraną. Sistemos žurnale to patvirtinti nepavyko.

Dabar abi pagrindinės programos tikrina kiekvieną paleidimą:

- **įvykio būsena** – kiekvienam kerneliui patikrinama `CL_EVENT_COMMAND_EXECUTION_STATUS`; nepavykusios komandos būsena neigiama:

```cpp
CL_CHECK(clGetEventInfo(e, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status),
                        &status, nullptr));
if (status != CL_COMPLETE) {
    std::fprintf(stderr, "GPU command failed: %s (status %d)\n", what, status);
    std::exit(1);
}
```

- **atrankinis patikrinimas** – `spot_check()` procesoriuje perskaičiuoja 4 C kampus ir 256 atsitiktinius C elementus; tam reikia tik ≈ 260·k operacijų, sekundės dalies net n = 14000;
- nepavykęs paleidimas baigiasi klaidos kodu, `bench.sh` praneša `FAILED`, eilutė **nepatenka į CSV**. Išbandyta su tyčia sugadintu kerneliu: `SPOT CHECK FAILED: matrix3[0][0] = 0, expected -115`.

`base.cpp` vis dar vykdo vieną ilgą kernelį, todėl su dideliu n gali būti nutrauktas – bet dabar pažymimas `FAILED` ir neišsaugomas. `opencl-opt` ilgų komandų išvengia: n = 14000 regblock atlieka 110 paleidimų per 5,8 s ir praeina patikrinimą.

**Profiliavimo keistenybė.** Grynasis kernelio laikas matuojamas profiliavimo eile: kiekvienas įvykis turi pradžios ir pabaigos laiko žymas (`CL_PROFILING_COMMAND_START` ir `..._END`). Pagal standartą tai nanosekundės, bet Apple realizacija grąžina `mach_absolute_time` taktus:

    1 taktas = 125 / 3 ns ≈ 41,67 ns   (Apple Silicon)

Todėl iš pradžių kernelio laikas išeidavo **lygiai 41,7 karto per mažas**. Dabar `event_seconds()` taktus perskaičiuoja pagal `mach_timebase_info` (`ns = ticks · numer / denom`). Grynasis laikas rodomas programos ataskaitoje, o lentelėse ir grafikuose naudojamas `mul`.

## 4. Apkrova vykdymo metu

Apkrova stebėta Bendrosios dalies 3 skyriuje aprašytu būdu, paleidus `opencl-opt` su regblock ir n = 16000 (trys matricos po ≈ 1 GB). Šiame paleidime daugyba truko 9,63 s, visa programa – 11,2 s.

![opencl-opt (regblock, n = 16000) daugybos metu btop lange: procesoriaus branduoliai beveik laisvi, programa naudoja 0,2 % procesoriaus, o GPU juosta rodo 100 %](docs/img/screens/opencl-regblock-btop.png)

![Tas pats momentas macmon lange: GPU apkrauta 100 % ties 1398 MHz ir naudoja ≈ 15,7 W, procesoriaus grupių apkrova maža](docs/img/screens/opencl-regblock-macmon.png)

![opencl-regblock šilumos žemėlapis: daugybos metu ištisai apkrauta tik GPU eilutė; ryškus P0 prieš daugybą – nuoseklus B pildymas, trumpi P branduolių blyksniai ≈ 7–9 s – ne šios programos](docs/img/load/opencl-regblock.png){: .wide }

Ką rodo šie vaizdai:

- **Daugybos metu vaizdo plokštė apkrauta 100 %.** Kas 100 ms įrašytų reikšmių medianos daugybos metu: GPU aktyvumas **100 %**, dažnis **1398 MHz** (prieš paleidimą – 444 MHz), galia **≈ 15,4 W** (prieš paleidimą – ≈ 0,1 W). Tą patį rodo abu ekrano vaizdai.
- **Procesorius beveik nedirba.** btop procesų sąraše `opencl-opt` naudoja **0,2 %** procesoriaus. Pagal `/usr/bin/time` visa programa sunaudojo 0,98 s naudotojo ir 0,30 s sistemos laiko per 11,19 s:

        (0,98 + 0,30) / 11,19 ≈ 0,11 branduolio

- **Tas procesoriaus laikas – B pildymas.** 0,98 s naudotojo laiko beveik sutampa su B pildymo trukme (0,90 s); šilumos žemėlapyje tai vienas ryškus P branduolys tarp A pildymo ir daugybos. Daugybos metu procesorius tik pateikia juostas ir laukia.
- **Kitas aktyvumas – ne šios programos.** E branduolius visą laiką apkrauna fono programos (Bendrosios dalies 2 skyrius), o daugybos viduryje (≈ 7–9 s) keli P branduoliai trumpam pakyla iki ≈ 3,4 GHz. Tikėtina, kad tai ekrano vaizdo fiksavimas (7-ąją sekundę) ar fono programos – todėl macmon P grupė rodo 17 %, nors programa naudoja 0,2 %.
- **Atmintis.** Naudojama RAM išauga ≈ 3 GB – tiek užima trys 16000 × 16000 matricos (3 · 1,02 GB); buferiai be kopijavimo, antros kopijos vaizdo plokštei nėra.

## 5. Laiko priklausomybė nuo matricų dydžio ir „branduolių skaičiaus“

### 5.1 Matricų dydis

`mul` laikas (vaizdo plokštės darbas ir sinchronizacija), n = k = p, kelių paleidimų medianos:

| n | naive | coalesced | tiled | regblock | regblock, palyginti su naive |
|---|-------|-----------|-------|----------|------------------------------|
| 2000 | 0,483 s | 0,070 s | 0,048 s | 0,020 s | 24× |
| 4000 | 3,87 s | 0,545 s | 0,375 s | 0,143 s | 27× |
| 8000 | 31,0 s | 4,39 s | 3,01 s | 1,19 s | 26× |
| 14000 | 168 s | 23,7 s | 16,1 s | 5,77 s | 29× |
| 16000 | – | 35,4 s | 24,0 s | 9,92 s | |

`naive` su n = 16000 nematuotas: vienas kernelis truktų ≈ 4 minutes ir greičiausiai būtų nutrauktas (3 skyrius). Laikas auga kaip n³ – padvigubinus n, ≈ 8 kartus (regblock 0,143 → 1,19 s).

Dar aiškiau tai rodo pralaidumas (n · k · p / mul):

| n | 2000 | 4000 | 8000 | 10000 | 14000 | 16000 |
|---|------|------|------|-------|-------|-------|
| naive, 10⁹ op/s | 16,6 | 16,6 | 16,5 | 16,4 | 16,3 | – |
| coalesced, 10⁹ op/s | 114 | 117 | 117 | 117 | 116 | 116 |
| tiled, 10⁹ op/s | 166 | 171 | 170 | 170 | 170 | 171 |
| regblock, 10⁹ op/s | 392 | 446 | 430 | 376 | 475 | 413 |

Grafikai rodo visos programos laiką (kartu su A ir B pildymu).

![Visas programos laikas pagal matricos dydį (abi ašys logaritminės): visos keturios tiesės lygiagrečios n³ tiesei, atstumas tarp naive ir regblock beveik pastovus](reports/ataskaita/img/ocl/plots/size.png)

![Tas pats pagal operacijų skaičių n·k·p tiesinėmis ašimis: kiekvieno kernelio taškai guli tiesėje be lūžių, jos posvyris – vienos operacijos kaina](reports/ataskaita/img/ocl/plots/size-linear.png)

- **Nėra staigaus lūžio**, kokį procesoriuje sukėlė TLB ties n ≈ 3547 (Bendrosios dalies 4.1 skyrius). Pirmieji trys kerneliai vienai operacijai skiria pastovų laiką (±2 %) visame intervale; net `naive` nesulėtėja didėjant n – jis visur vienodai lėtas.
- **Pagreitėjimas naive → regblock išlieka 24–29×** visuose dydžiuose. Vadinasi, `naive` problema buvo ne dydis, o **kreipinių į atmintį tvarka**: ją pataisius, nauda vienoda ir mažoms, ir didelėms matricoms.
- **regblock pralaidumas svyruoja (376–475·10⁹ op/s).** Esant n = 2000 daugyba trunka tik 0,02 s, todėl pastoviosios sąnaudos (≈ 3 ms) sudaro pastebimą dalį. Didesnių dydžių svyravimų priežasties atskirai netyrėme; tikėtina, kad įtakos turi fono apkrova ir tai, kad ta pati vaizdo plokštė tuo metu piešia ekraną.

### 5.2 „Branduolių skaičius“ vaizdo plokštėje

Užduoties reikalavimas (a) – laiko priklausomybė nuo branduolių skaičiaus – vaizdo plokštei tiesiogiai netaikomas. Įrenginys turi **19 skaičiavimo vienetų** (`CL_DEVICE_MAX_COMPUTE_UNITS`), ir programa šio skaičiaus nekeičia: OpenCL pats paskirsto darbo grupes visiems vienetams (todėl CSV „gijų“ stulpelyje visada 19). Lygiagretumą čia lemia **darbo elementų ir darbo grupių skaičius** – NDRange ir darbo grupės dydis.

n = 4000 `naive`, `coalesced` ir `tiled` paleidžia po 16·10⁶ darbo elementų (vieną kiekvienam C elementui), o regblock – tik 4 juostas po 65 536, t. y. **≈ 60 kartų mažiau**, bet yra 27 kartus greitesnis už `naive`. Viename regblock paleidime yra 256 darbo grupės – vidutiniškai ≈ 13 kiekvienam iš 19 skaičiavimo vienetų. Daugiau darbo elementų nebūtinai geriau.

Kaip lygiagretumo dydis veikia laiką, rodo derinimo lentelė (2.4 skyrius): sumažinus darbo grupę nuo 16 × 16 iki **8 × 8** (64 darbo elementai vietoj 256), laikas pailgėja nuo 0,143 iki **0,201 s**. Tikėtina, kad mažoje grupėje per mažai darbo elementų atminties delsai paslėpti (kol vieni laukia duomenų, kiti skaičiuoja), o kiekvienas lokalusis blokas panaudojamas mažiau kartų.

## 6. Vaizdo plokštė ir procesorius; kas lieka

Palyginimas su optimizuotais procesoriaus variantais, n = 4000 (medianos):

| variantas | mul | total |
|-----------|-----|-------|
| `seq-opt` (1 branduolys) | 1,58 s | 1,66 s |
| `openmp-opt2` (8 gijos) | 0,218 s | 0,276 s |
| `mpi-opt` (8 procesai) | 0,242 s | 0,300 s |
| **`opencl-opt` regblock** | **0,143 s** | **0,220 s** |

Daugybą vaizdo plokštė atlieka 1,5 karto greičiau nei `openmp-opt2` ir 11 kartų greičiau nei vienas procesoriaus branduolys su tuo pačiu atminties optimizavimu. Visos programos skirtumas mažesnis (0,276 → 0,220 s, 1,25×), nes kitos fazės naudos iš vaizdo plokštės negauna:

- **Nuoseklus B pildymas.** n = 4000 jis trunka 0,058 s ir sudaro **27 %** viso regblock laiko (0,058 / 0,220). Net jei daugyba truktų 0 s, programa negalėtų būti trumpesnė už A ir B pildymą (0,076 s). B pildymas auga kaip n², daugyba – kaip n³, todėl didėjant matricoms jo dalis mažėja (Gustafsono dėsnis, Bendrosios dalies 5 skyrius): n = 16000 – jau tik 8 % (0,86 / 11,06 s). Ten ir atotrūkis nuo procesoriaus didžiausias: visa programa trunka 11,1 s, geriausias `openmp-opt2` (12 gijų) – 16,3 s, `seq-opt` – 107 s.
- **A pildymas vaizdo plokštėje nėra greitesnis.** n = 4000 jis trunka 0,017 s, `openmp-opt2` su 8 gijomis – 0,002 s. Tikėtina, kad vyrauja ne skaičiavimas, o buferio kūrimo, paleidimo ir sinchronizacijos sąnaudos.
- **Sveikųjų skaičių aritmetika.** Matricos yra int32, o vaizdo plokštė sveikųjų skaičių daugybą atlieka lėčiau nei FP32. Šiame uždavinyje `float` sandaugos ir sumos būtų tikslios, kol |sum| < 2²⁴ (x = 10 ir k ≤ 1,6 milijono), todėl kernelis su `float` greičiausiai būtų greitesnis. To nedarėme, nes tai pakeistų užduoties duomenų tipą. Kitos neįgyvendintos idėjos – vektoriniai įkėlimai (`int4`) ir dvigubas lokaliųjų blokų buferizavimas.

## Išvados

1. **Užduotis įgyvendinta vaizdo plokštėje:** A pildoma lygiagrečiai kerneliu `fill_matrix1`, B – nuosekliai procesoriuje, C = A × B – lygiagrečiai kerneliu `matmul`; visi kerneliai praeina `--verify`, o kiekvienas matavimo paleidimas – atrankinį patikrinimą.
2. **Vaizdo plokštės greitį lemia kreipinių į atmintį tvarka.** Ta pati aparatinė įranga, keičiant tik kernelį, atlieka nuo 16·10⁹ iki ≈ 450·10⁹ operacijų per sekundę (27×). Didžiausią žingsnį (7×) davė vien teisinga NDRange dimensija – sujungti kreipiniai; lokalioji atmintis davė 1,45×, blokavimas registruose – 2,6×.
3. **Laikas auga kaip n³ be lūžių;** pralaidumas beveik nepriklauso nuo dydžio, naive → regblock pagreitėjimas visuose dydžiuose 24–29×. Skaičiavimo vienetų skaičius fiksuotas (19), lygiagretumą lemia NDRange ir darbo grupės dydis: per maža darbo grupė (8 × 8) sulėtina 40 %.
4. **Apkrova:** daugybos metu vaizdo plokštė apkrauta 100 % (1398 MHz, ≈ 15 W), o programa naudoja vidutiniškai tik 0,11 procesoriaus branduolio – procesorius dirba beveik tik nuosekliai pildydamas B.
5. **Vaizdo plokštės klaidos gali būti tylios.** Nutrauktas kernelis „baigėsi“ per 0,22 s vietoj 168 s, ir netikras laikas pateko į rezultatus. Būtini įvykio būsenos ir rezultato patikrinimai, o ilgus darbus reikia skaidyti į trumpus paleidimus.
6. **Vaizdo plokštė – greičiausias variantas** (n = 4000 daugyba 0,143 s, `openmp-opt2` – 0,218 s; n = 16000 visa programa 11,1 s, palyginti su 16,3 s), tačiau nuoseklus B pildymas n = 4000 jau sudaro 27 % viso laiko, todėl tolesnis kernelio spartinimas visą programą pagreitintų vis mažiau.
