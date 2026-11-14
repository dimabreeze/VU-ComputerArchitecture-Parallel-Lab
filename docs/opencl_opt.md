# OpenCL opt: nuo 16 iki ~450 milijardų operacijų per sekundę

Pagrindinė programa (host): [`src/opencl/opt.cpp`](../src/opencl/opt.cpp), tikslas `opencl-opt`.
Kerneliai, po vieną kiekvienam optimizavimo žingsniui: [`src/opencl/opt/`](../src/opencl/opt/) —
`coalesced.cl`, `tiled.cl`, `regblock.cl`. Bendras kodas (generatorius, `fill_matrix1`,
makrokomanda `MAD`): [`src/opencl/common.cl`](../src/opencl/common.cl), kurį pagrindinė programa
prijungia kernelio failo pradžioje.

Pradinis variantas nepakeistas: `opencl-base` su `src/opencl/base/naive.cl`.
`make bench` paleidžia kiekvieną programos kernelį: `src/opencl/<X>/*.cl` programai
`build/opencl-<X>`.

Užduoties semantika tokia pati kaip base:

- matrix1 pildoma lygiagrečiai GPU;
- matrix2 pildoma nuosekliai CPU;
- daugyba vykdoma lygiagrečiai GPU.

Visi kerneliai sėkmingai praeina `--verify` su nesulygiuotais dydžiais (`37x51x29`, `1x1x1`, `100x3x77`,
`129x257x130`, `1001x999x1003`), įskaitant atvejus, kai skaičiavimas padalytas į daug paleidimų.

Kompiuteris: Apple M2 Pro, GPU su 19 branduolių (Metal 3), Apple OpenCL 1.2. Apple
OpenCL yra pasenęs (deprecated), bet vis dar veikia.

## 1. Atspirties taškas

`naive.cl` pasiekia apie **16·10⁹** daugybos ir sudėties operacijų per sekundę esant bet kokiam dydžiui
(n = 4000: 64·10⁹ / 3,87 s; n = 14000: 2,7·10¹² / 168 s). Tai 18× lėčiau nei
CPU (`openmp-opt2`, 8 gijos) ir maža dalis to, ką gali GPU
(apie 6,8·10¹² FP32 operacijų per sekundę).

## 2. Kerneliai

### 2.1 coalesced.cl: tinkama NDRange dimensija

OpenCL kaimyniniai darbo elementai (work-items) skiriasi **0 dimensija**. `naive.cl` naudoja
`i = get_global_id(0)`, todėl kaimyniniai darbo elementai:
- skaito `m1[i*k + z]` žingsniu k, po vieną podėlio eilutę kiekvienam darbo elementui;
- rašo `m3[i*p + j]` žingsniu p.

`coalesced.cl` stulpelį perkelia į 0 dimensiją (`j = get_global_id(0)`). Tada kaimynai
viena transakcija skaito iš eilės einančius `m2[z*p + j]`, o visi jie skaito tą patį
`m1[i*k + z]` (transliavimas).

Jame taip pat naudojama `mad24()`: sveikųjų skaičių daugybos ir sudėties operacija operandams, telpantiems į 24 bitus,
kuri vykdoma greitaisiais GPU blokais. Pagrindinė programa ją įjungia (`-DUSE_MAD24`) tik kai
`x < 2²³`. matrix1 reikšmės yra intervale [1, x], o matrix2 reikšmės yra ±1, todėl rezultatas yra tas pats
32 bitų sveikasis skaičius. Didesniems `x` naudojamas paprastas `a*b + c`, ir šis kelias taip pat
patikrintas.

Darbo grupės (work-group) dydis fiksuotas: 16 × 16.

### 2.2 tiled.cl: lokalioji atmintis

16 × 16 darbo grupė apskaičiuoja 16 × 16 C bloką. Kiekviename 16 ilgio žingsnyje išilgai k
grupė įkelia po 16 × 16 matrix1 ir matrix2 bloką (tile) į `__local` (luste esančią)
atmintį, po vieną elementą kiekvienam darbo elementui, ir sinchronizuojasi su `barrier`. Tada kiekvienas įkeltas
elementas panaudojamas 16 kartų, todėl globaliosios atminties srautas sumažėja 16 kartų. Už ribų esantys
elementai įkeliami kaip 0, todėl kraštams nereikia specialaus kodo.

### 2.3 regblock.cl: blokavimas registruose

Kiekvienas darbo elementas apskaičiuoja **8 × 8** C bloką ir laiko jį registruose. 16 × 16
darbo grupė apskaičiuoja 128 × 128 bloką. Kiekviename k žingsnyje darbo elementas nuskaito 8 + 8
reikšmes iš lokaliosios atminties ir atlieka 64 daugybos ir sudėties operacijas; `tiled.cl` reikėjo 2 nuskaitymų
vienai daugybos ir sudėties operacijai. Tai CPU mikrobranduolio iš
`include/gemm.hpp` atitikmuo GPU.

Darbo elementui (lx, ly) priklauso eilutės `ly + 16·r` ir stulpeliai `lx + 16·c` (su žingsniu).
Todėl kaimyniniai darbo elementai skaito kaimyninius lokaliojo bloko stulpelius be
bankų konfliktų ir rašo į kaimyninius adresus. Dėl tos pačios priežasties lokalusis matrix1 blokas
papildytas vienu stulpeliu (`[TM][TK+1]`).

Parametrai yra makrokomandos: `WM`, `WN` (blokas vienam darbo elementui), `LX`, `LY` (darbo grupė),
`TK` (bloko gylis). Jie suderinti esant n = 4000 (`mul`, geriausias iš 3) neredaguojant
failo, per aplinkos kintamąjį `CL_BUILD_OPTS`. Tai vienkartinis derinimo eksperimentas (rankiniai
paleidimai, ne `make bench` medianos), todėl jo skaičiai gali šiek tiek skirtis nuo 4 skyriaus:

| konfigūracija (WM × WN, darbo grupė, TK) | n = 4000 | n = 8000 |
|------------------------------------------|----------|----------|
| 4 × 4, 16 × 16, 16                       | 0,157 s  | 1,264 s  |
| 4 × 8, 16 × 16, 16                       | 0,153 s  |          |
| 8 × 4, 16 × 16, 16                       | 0,145 s  | 1,199 s  |
| **8 × 8, 16 × 16, 16 (pasirinkta)**      | **0,143 s** | **1,184 s** |
| 8 × 8, 8 × 8, 16                         | 0,201 s  |          |
| 8 × 8, 16 × 16, 8                        | 0,145 s  |          |
| 4 × 4, 16 × 16, 32                       | 0,171 s  |          |
| 8 × 4, 32 × 8, 16                        | 0,149 s  |          |

```sh
CL_BUILD_OPTS="-DWM=8 -DWN=4" build/opencl-opt -s 4000 --kernel src/opencl/opt/regblock.cl
```

## 3. Pagrindinė programa (opt.cpp)

- **Buferiai be kopijavimo (zero-copy).** Apple Silicon turi bendrąją CPU ir GPU atmintį. Buferiai kuriami su
  `CL_MEM_USE_HOST_PTR` ant pagrindinės programos matricų, o `clEnqueueMapBuffer` /
  `Unmap` padaro GPU rezultatus matomus pagrindinei programai jų nekopijuojant. Vieno
  profiliuoto regblock paleidimo (n = 4000) `mul` fazė truko 0,191 s, iš jų pats kernelis
  – 0,187 s: buferių paruošimas ir sinchronizacija trunka apie 3 ms (`make bench` mediana –
  0,143 s, 4 skyrius). `base.cpp` kopijuoja 2 × 64 MB.
- **Save aprašantys kerneliai.** Kiekviename kernelio faile yra kernelis `geometry`, kuris
  grąžina `{LX, LY, TM, TN}`: darbo grupės dydį ir vienai grupei tenkantį C bloką. Viena
  pagrindinė programa aptarnauja visus kernelius, o globalūs dydžiai suapvalinami aukštyn iki sveikų blokų.
- **Paleidimas eilučių juostomis.** Daugyba padalijama į paleidimus, kurių kiekvienas atlieka ne daugiau kaip
  2³⁴ ≈ 1,7·10¹⁰ daugybos ir sudėties operacijų, sveikais eilučių blokais (`BAND_OPS`; n = 4000:
  4 paleidimai, n = 14000: 110). Kiekvienas paleidimas yra trumpas.
- **Kiekvienas programos paleidimas tikrinamas.**
  - Baigus patikrinama kiekvieno kernelio įvykio būsena. Įrenginyje nepavykusios komandos
    būsena yra neigiama.
  - `spot_check()` iš `common.cpp` CPU perskaičiuoja 256 atsitiktinius matrix3 elementus ir
    kampus. Tam reikia 256·k operacijų, t. y. sekundės dalį net
    kai n = 14000.
  - Nepavykęs paleidimas išveda klaidą ir baigiasi kodu 3, todėl `bench.sh` praneša
    `FAILED`, o eilutė nepatenka į CSV. Išbandyta su tyčia sugadintu
    kerneliu: `SPOT CHECK FAILED: matrix3[0][0] = 0, expected -115`.
- **Profiliavimas.** Ataskaitoje rodomas grynasis kernelio laikas (visų paleidimų suma). CSV
  jo neįtraukia.

### Kodėl svarbūs patikrinimai ir juostos

Su `opencl-base`, kai n = 14000 (3 pakartojimai), 1-asis ir 3-iasis paleidimai truko 168 s, o 2-asis paleidimas
„truko“ 0,22 s. Tai būtų 2,7·10¹² / 0,22 ≈ 12·10¹² operacijų per sekundę, daugiau
nei GPU FP32 didžiausias našumas, todėl kernelis negalėjo atlikti šio darbo. Greičiausiai jis
buvo nutrauktas arba atmestas. Tuo metu `base.cpp` netikrino kernelio būsenos,
o blokuojantis rezultato buferio skaitymas vis tiek grąžina sėkmę, todėl neteisingas rezultatas
buvo pateiktas su netikru laiku. Dabar `base.cpp` turi tuos pačius patikrinimus kaip `opt.cpp`:
įvykio būseną ir atrankinį patikrinimą. Jis vis dar vykdo vieną ilgą kernelį be juostų, todėl toks
paleidimas vis dar gali būti nutrauktas, tačiau jis pažymimas kaip `FAILED` ir neišsaugomas. Tikėtina priežastis – macOS apsauga nuo ilgų GPU komandų:
vienas kernelis veikė beveik 3 minutes tame pačiame GPU, kuris valdo ekraną. To
nepavyko patvirtinti iš sistemos žurnalo.

`opencl-opt` išvengia ilgų komandų (juostos) ir aptinka tokius gedimus (būsena ir
atrankinis patikrinimas). Kai n = 14000, regblock atlieka 110 paleidimų per 5,77 s (mediana) ir praeina
patikrinimą.

### Apple OpenCL profiliavimo keistenybė

Pagal standartą `CL_PROFILING_COMMAND_START/END` turi būti nanosekundėmis.
Apple realizacija vietoj jų grąžina `mach_absolute_time` taktus
(Apple Silicon 1 taktas = 125/3 ≈ 41,67 ns). Prieš pataisymą kernelio laikas
išeidavo lygiai 41,7 karto per mažas. `event_seconds()` konvertuoja reikšmę naudodama
`mach_timebase_info`.

## 4. Rezultatai

`mul` fazė (GPU darbas + sinchronizacija), sekundės – `make bench` medianos (`results/csv/`):

| n     | naive (base) | coalesced | tiled  | regblock | regblock, palyginti su naive |
|-------|--------------|-----------|--------|----------|------------------------------|
| 2000  | 0,483        | 0,070     | 0,048  | 0,020    | 24×                          |
| 4000  | 3,87         | 0,545     | 0,375  | 0,143    | 27×                          |
| 8000  | 31,0         | 4,39      | 3,01   | 1,19     | 26×                          |
| 14000 | 168          | 23,7      | 16,1   | 5,77     | 29×                          |
| 16000 | –            | 35,4      | 24,0   | 9,93     |                              |

Kiekvienas žingsnis duoda savo naudą (n = 4000): sujungti kreipiniai (coalescing) – 7,1×, lokalioji
atmintis – 1,45×, blokavimas registruose – 2,6×.

regblock našumas – nuo 392·10⁹ (n = 2000) iki 475·10⁹ (n = 14000) daugybos ir sudėties operacijų
per sekundę; kai n = 4000, 64·10⁹ / 0,143 s ≈ **446·10⁹**. Palyginimui naive – ≈ 16·10⁹.

Palyginimas su CPU, kai n = 4000 (`mul`, medianos):

| variantas                    | mul      |
|------------------------------|----------|
| seq-opt (1 branduolys)       | 1,58 s   |
| openmp-opt2 (8 gijos)        | 0,218 s  |
| mpi-opt (8 procesai)         | 0,242 s  |
| **opencl-opt regblock (GPU)** | **0,143 s** |

## 5. Likę apribojimai ir idėjos (neįgyvendintos)

- **fill2.** Nuoseklus matrix2 generavimas CPU (0,058 s, kai n = 4000) dabar sudaro
  27 % viso laiko (0,220 s). Jis nuoseklus pagal užduoties apibrėžimą.
- **int32 GPU.** Sveikųjų skaičių aritmetika GPU yra lėtesnė nei FP32. Čia
  sandaugos ir sumos būtų tikslios `float` tipe, kol |sum| < 2²⁴, o tai galioja,
  kai x = 10 ir k ≤ 1,6 milijono, todėl kernelis su float FMA tikriausiai būtų greitesnis.
  Tai neatlikta, nes pakeistų užduoties duomenų tipą.
- Vektoriniai įkėlimai (`int4`), dvigubas lokaliųjų blokų buferizavimas ir tolesnis derinimas kiekvienam
  dydžiui.
