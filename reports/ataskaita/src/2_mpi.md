title: MPI: procesai su atskira atmintimi
subtitle: mpi-base → mpi-opt
part: 2 dalis

# 2 dalis. MPI

!!! fact "Svarbiausi faktai"
    - Ta pati užduotis, kitas skaičiavimo branduolys: `mpi-opt` už kurso šabloną (`mpi-base`) greitesnis **126 kartus su 1 procesu ir 98 kartus su 8 procesais** (n = 4000: 29,3 s → 0,300 s) (2 ir 5 skyriai).
    - 8 procesai užima **6,96 branduolio**, iš jų ≈ 10 % procesoriaus laiko – sistemos darbas (MPI perdavimas per bendrai naudojamą atmintį, atminties valdymas) (3 skyrius).
    - Kiekvienas procesas laiko **visą B kopiją**: n = 14000 tai 8 × 784 MB ≈ 6,3 GB. macOS praneša apie atminties stygių, auga keitimo sritis, o tas pats paleidimas trunka **23,3 s arba 16,3 s** – priklausomai nuo to, kiek atminties laisva prieš paleidimą (3 skyrius).
    - Pagreitėjimas su 8 procesais didžiausias vidutinėms matricoms – **6,2× (n = 8000)** – ir krenta iki **4,6× (n = 16000)**; su 12 procesų n = 16000 lieka tik **2,3×**: 12 procesų visada lėtesni už 8 (4 skyrius).
    - Kai n ≥ 6000, iki 4 procesų pagreitėjimą visiškai paaiškina Amdalo dėsnis (e ≈ f ≈ 1–2 %); didelėms matricoms Karpo–Flato metrika auga su procesų skaičiumi – **10 % su 8 ir 38 % su 12 procesų** (n = 16000) (6 skyrius).

## 1. Kaip užduotis įgyvendinta su MPI

MPI programą sudaro keli **procesai**, kurių kiekvienas turi savo atmintį; duomenimis jie keičiasi tik siųsdami pranešimus. Programa paleidžiama `mpirun -np 8 build/mpi-opt -s 14000` – visi 8 procesai vykdo tą patį kodą, o elgesį skiria proceso numeris – **rangas** (0…N−1). Viename kompiuteryje Open MPI pranešimus perduoda per bendrai naudojamą atmintį: tai atminties kopijavimas, kurį atlieka patys procesai.

Užduoties reikalavimai įgyvendinti taip:

1. **Matrica A pildoma lygiagrečiai:** kiekvienas procesas pildo tik jam priskirtas A eilutes. Elementas yra deterministinė funkcija nuo sėklos ir globalaus indekso (Bendrosios dalies 2.3 skyrius), todėl rezultatas nepriklauso nuo procesų skaičiaus.
2. **Matrica B pildoma nuosekliai** – tik rango 0 procese. B reikia visiems procesams, todėl ji **transliuojama** (`mpi-base` – `MPI_Bcast`, `mpi-opt` – `MPI_Ibcast` ir `MPI_Wait`, kurie be konvejerio atlieka tą patį vieną transliavimą) – po transliavimo kiekvienas procesas turi visą B kopiją.
3. **Daugyba lygiagreti:** kiekvienas procesas skaičiuoja C eilutes, atitinkančias jo A eilutes (C eilutei reikia vienos A eilutės ir visos B).
4. **Rezultatas surenkamas** rango 0 procese: `mpi-base` – atskirais pranešimais, `mpi-opt` – vienu **surinkimu** (`MPI_Gatherv`). Rangas 0 spausdina ataskaitą ir, jei nurodyta `--verify`, palygina C su nuosekliu skaičiavimu.

| MPI funkcija | paskirtis |
|--------------|-----------|
| `MPI_Init`, `MPI_Finalize`, `MPI_Comm_size`, `MPI_Comm_rank` | MPI pradžia ir pabaiga, procesų skaičius ir savo rangas |
| `MPI_Barrier`, `MPI_Wtime` | fazių pradžios sinchronizavimas ir laiko matavimas |
| `MPI_Bcast` | `mpi-base`: B transliavimas iš rango 0; abu variantai – atsitiktinių skaičių sėklos transliavimas |
| `MPI_Send`, `MPI_Recv` | `mpi-base`: kiekviena C eilutė siunčiama rangui 0 atskiru pranešimu |
| `MPI_Gatherv` su `MPI_IN_PLACE` | `mpi-opt`: visų C blokų surinkimas vienu kolektyviniu kvietimu |
| `MPI_Ibcast`, `MPI_Wait`, `MPI_Test` | `mpi-opt`: B transliavimas – numatytuoju atveju viena dalimi, su konvejeriu – dalimis (`MPI_Test` varo perdavimą) |

Fazės matuojamos kaip ir kitose programose (`fill1`, `fill2`, `mul`, `total`, Bendrosios dalies 2.3 skyrius). Prieš kiekvieną fazę procesai sinchronizuojami `MPI_Barrier`, todėl fazės laikas – lėčiausio proceso laikas.

## 2. Programos: mpi-base ir mpi-opt

### 2.1 mpi-base – kurso šablonas, pataisytas

`mpi-base` (`src/mpi/base.cpp`) – kurso šablonas „MPI program sample“, pataisytas taip, kad veiktų teisingai. **Algoritmas nepakeistas:** eilutės skirstomos cikliškai (`i = rank, rank + size, …`), kiekviena C eilutė skaičiuojama vadovėliniu i-j-z ciklu ir siunčiama rangui 0 su `MPI_Send` / `MPI_Recv`.

```cpp
for (int i = rank; i < a.n; i += size)          // cikliškas eilučių paskirstymas
    for (int j = 0; j < a.p; ++j) {
        int sum = 0;
        for (int z = 0; z < a.k; ++z) sum += m1.at(i, z) * m2.at(z, j);
        m3.at(i, j) = sum;
    }
if (rank != 0)                                   // kiekviena eilutė – atskiras pranešimas
    for (int i = rank; i < a.n; i += size)
        MPI_Send(m3.row(i), a.p, MPI_INT, 0, TAG_M3, MPI_COMM_WORLD);
else
    for (int src = 1; src < size; ++src)
        for (int i = src; i < a.n; i += size)
            MPI_Recv(m3.row(i), a.p, MPI_INT, src, TAG_M3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```

Kas pataisyta, palyginti su šablonu:

- **Duomenys pagal užduotį.** Šablone kiekvienas procesas generavo savo matricas su `rand()`, todėl procesai daugino *skirtingus* duomenis. Dabar A pildoma lygiagrečiai (kiekvienas procesas – savo eilutes), B – nuosekliai rango 0 procese ir transliuojama `MPI_Bcast`; visuose procesuose duomenys vienodi.
- **Pranešimų žymės.** Šablonas eilutei i naudojo žymę `10 + i`, kuri dideliems n viršija leistiną didžiausią žymę `MPI_TAG_UB`. Dabar kiekvienam srautui naudojama viena žymė: MPI garantuoja, kad pranešimai tarp tos pačios procesų poros ateina siuntimo tvarka.
- **Klaidos kode:** perteklinis `}` prieš `MPI_Finalize`; laiką spausdino kiekvienas procesas – dabar ataskaitą spausdina tik rangas 0.

Pridėta: komandinės eilutės argumentai, deterministiniai generatoriai, rezultato tikrinimas (`--verify`) ir CSV ataskaita. `mpi-base` fazė `fill2` apima ir B transliavimą.

### 2.2 mpi-opt – greitas skaičiavimo branduolys ir vienas surinkimas

`mpi-opt` (`src/mpi/opt.cpp`) išlaiko tą pačią užduoties semantiką, bet keičia tris dalykus.

**1. Skaičiavimo branduolys iš `include/gemm.hpp`** – tas pats blokinis, pakuojantis, registrų mikrobranduolį naudojantis kodas kaip `seq-opt` (Bendrosios dalies 4.3 skyrius). Vadovėlinis i-j-z ciklas skaito B stulpeliais ir, kai n > 3547, atsiremia į TLB ribą (Bendrosios dalies 4.1 skyrius); `gemm.hpp` šios problemos neturi. Tai pagrindinis laimėjimas.

**2. Ištisiniai eilučių blokai vietoj ciklinio paskirstymo.** Procesas r gauna `n / size` eilučių, o pirmieji `n % size` procesų – viena daugiau. Procesas saugo tik savo A ir C eilutes (rangas 0 – pilnas, nes jam vis tiek reikia viso rezultato), o C surenkama **vienu** `MPI_Gatherv`. Rangas 0 naudoja `MPI_IN_PLACE` – jo paties blokas nekopijuojamas.

```cpp
for (int r = 0, off = 0; r < size; ++r) {       // blokinis paskirstymas
    rows[r] = a.n / size + (r < a.n % size ? 1 : 0);
    offs[r] = off;  off += rows[r];
}
// ... daugyba: gemm_add(my_rows, ...) su savo eilučių bloku ...
if (rank == 0)                                   // vienas surinkimas vietoj n pranešimų
    MPI_Gatherv(MPI_IN_PLACE, 0, MPI_INT, m3.data_ptr(), cnt.data(), dsp.data(),
                MPI_INT, 0, MPI_COMM_WORLD);
else
    MPI_Gatherv(m3.data_ptr(), cnt[rank], MPI_INT, nullptr, nullptr, nullptr,
                MPI_INT, 0, MPI_COMM_WORLD);
```

**3. Transliavimas įskaitomas į daugybą.** B transliavimas (`MPI_Ibcast` + `MPI_Wait`; be konvejerio – viena dalis) perkeltas į `mul` fazę, todėl `mpi-opt` fazė `fill2` – tik grynas nuoseklus B pildymas. Tai reikalinga Amdalo analizei (6 skyrius).

!!! note "Konvejeris su MPI_Ibcast: realizuotas, bet išjungtas"
    Kompiliuojant su `-DKC=<eilučių>` B transliuojama dalimis neblokuojančiu `MPI_Ibcast`: procesas daugina su dalimi c, kol perduodamos dalys c+1… Viename kompiuteryje tai **nepadėjo** (n = 4000, 8 procesai: `mul` 0,243 s be konvejerio, 0,263 s su KC = 512, 0,326 s su KC = 128). Slėpti beveik nėra ką – 64 MB transliavimas trunka ≈ 15 ms, ≈ 6 % daugybos laiko. Be to, perdavimas per bendrai naudojamą atmintį – tai pačių procesų atliekamas kopijavimas, o Open MPI neblokuojančias operacijas varo į priekį tik MPI kvietimų viduje: **asinchroninės pažangos nėra**, ir laukimas `MPI_Wait` su konvejeriu net išaugo. Kodas paliktas paleidimams klasteryje.

## 3. Branduolių apkrova vykdymo metu

Apkrova stebėta Bendrosios dalies 3 skyriuje aprašytais būdais; scenarijus – `mpi-opt`, n = 14000, 8 procesai. Paleidimo trukmė pagal `/usr/bin/time` – 16,3 s; tai `mpirun` laikas su MPI paleidimu ir atminties atlaisvinimu, pačios programos `total` – 15,3 s (pagrindinių matavimų mediana – 15,34 s).

![mpi-opt (n = 14000, 8 procesai) vykdymo metu: 8 atskiri procesai po 2 gijas, kiekvienas ≈ 88–95 % vieno branduolio; P branduoliai C4–C11 užimti 100 %, bet ir E branduoliai C0–C3 – 46–67 %](docs/img/screens/mpi-opt-np8-btop.png)

**Ką rodo ekrano vaizdas.** Skirtingai nei gijos, kiekvienas MPI procesas procesų sąraše matomas **atskira eilute** su savo numeriu (PID); kiekvienas turi 2 gijas (antroji, tikėtina, – pagalbinė Open MPI gija) ir užima ≈ 88–95 % vieno branduolio. Visi 8 našieji branduoliai (C4–C11) užimti 100 %. Taupieji branduoliai (C0–C3) mūsų procesų neturi, tačiau užimti 46–67 % – gerokai daugiau nei fono apkrova prieš paleidimą.

**Kiek branduolių užėmė programa.** `/usr/bin/time` duomenimis 8 procesai sunaudojo 101,7 s vartotojo (user) ir **11,8 s sistemos (sys)** procesoriaus laiko – iš viso 113,5 s per 16,3 s, t. y. **vidutiniškai 6,96 branduolio**. Sistemos laikas sudaro ≈ 10 % viso procesoriaus laiko: tai operacinės sistemos darbas procesų naudai – MPI perdavimas per bendrai naudojamą atmintį ir atminties valdymas (puslapių išskyrimas, glaudinimas). Vienos gijos programose sistemos laikas sudaro tik ≈ 2 % (`seq-base`: 0,54 s iš 26,8 s).

![mpi-opt šilumos žemėlapis (n = 14000, 8 procesai): visi P branduoliai užimti visą daugybos laiką, E branduoliai dirba 40–80 % su pliūpsniais iki 100 %, naudojama RAM šokteli nuo ≈ 9,7 iki ≈ 13,3 GB, o keitimo sritis daugybos pabaigoje auga](docs/img/load/mpi-opt-np8.png){: .wide }

### 3.1 Atminties stygius: kiekvienas procesas turi savo B kopiją

!!! fact "Atminties stygius – MPI kaina viename kompiuteryje"
    Procesai neturi bendros atminties, todėl **kiekvienas procesas laiko visą B kopiją**: n = 14000 tai 784 MB vienam procesui ir **≈ 6,3 GB 8 procesams**; n = 16000 – po 1 GB (8,2 GB su 8 procesais, 12,3 GB su 12). Kartu su A ir C blokais bei pilnomis rango 0 matricomis n = 14000 programai reikia ≈ 9 GB iš 16 GB, kai kitos programos jau užima 7–10 GB.

    Paleidimo metu macOS atminties stygiaus rodiklis pakilo nuo 34 % iki **68–71 % (lygis 1 – „įspėjimas“)**, augo keitimo sritis. Tas pats paleidimas truko **23,3 s, kai prieš paleidimą buvo laisva ≈ 5 GB, ir 16,3 s, kai ≈ 6,5 GB**.

Apkrova stebėta du kartus. Pirmojo paleidimo duomenys užrašyti jo metu, antrojo – išsaugoti (`results/monitor/mpi-opt-np8.json`, `.jsonl`) ir parodyti šilumos žemėlapyje.

| paleidimas | laisva RAM prieš | trukmė | keitimo srities prieaugis | atminties stygius |
|------------|------------------|--------|---------------------------|-------------------|
| pirmasis | ≈ 5 GB | **23,3 s** | **+2,1 GB** | iki 71 %, lygis 1 |
| antrasis (žemėlapyje) | ≈ 6,5 GB | **16,3 s** | +0,3 GB (3,03 → 3,29 GB) | iki 68 %, lygis 1 |

Antrojo paleidimo įrašuose (kas 100 ms) matyti visa seka. Per pirmąją sekundę naudojama RAM šokteli nuo 9,7 iki 13,3 GB, **po 3,2 s** stygius pasiekia 1 lygį ir jame lieka iki pabaigos; daugybos pabaigoje (12,8–13,4 s ir 15,9–16,1 s) auga keitimo sritis. Prieš paleidimą E branduoliai užimti ≈ 25 % (fonas), daugybos metu – **≈ 55 %**, su pliūpsniais iki 80–100 % (ties ≈ 8 s ir 12–13 s). Mūsų procesai tuo metu dirba P branduoliuose (P0–P7 – vidutiniškai 98 %), todėl papildoma E branduolių apkrova – **sistemos darbas**, greičiausiai atminties glaudinimas ir keitimo srities rašymas.

**RAM apskaita nesutampa su poreikiu.** Procesai išskiria ≈ 9 GB, o naudojama RAM padidėja tik ≈ 3,6 GB, ir `btop` kiekvienam procesui rodo ≈ 370–390 MB, nors vien B kopija užima 784 MB. Tikėtinas paaiškinimas – macOS atminties glaudinimas: B sudaryta tik iš reikšmių 1 ir −1, todėl suspaudžiama labai gerai, o suspausti puslapiai į proceso atmintį neįskaitomi. Atskirai to nematavome, bet tai dera su ≈ 10 % sistemos laiko ir apkrautais E branduoliais.

**Ryšys su matavimais.** Bendras B kopijų dydis paaiškina, kur krenta plečiamumas (4 skyrius):

| n | B vienam procesui | 8 procesai: visos B kopijos | pagreitėjimas | 12 procesų: visos B kopijos | pagreitėjimas |
|---|-------------------|-----------------------------|---------------|-----------------------------|---------------|
| 8000 | 256 MB | 2,0 GB | 6,2× | 3,1 GB | 5,7× |
| 10000 | 400 MB | 3,2 GB | 5,8× | 4,8 GB | 4,6× |
| 12000 | 576 MB | 4,6 GB | 5,7× | **6,9 GB** | **2,9×** |
| 14000 | 784 MB | **6,3 GB** | **4,8×** | 9,4 GB | 2,6× |
| 16000 | 1024 MB | **8,2 GB** | **4,6×** | 12,3 GB | 2,3× |

Pagreitėjimas staigiai krenta tada, kai visos B kopijos kartu viršija ≈ 6 GB – maždaug tiek, kiek laisvos atminties būna prieš paleidimą (Bendrosios dalies 2 skyrius). Tai sutapimas, o ne įrodymas, tačiau abu stebėjimai – apkrovos įrašai ir pagreitėjimo kritimas – rodo tą pačią priežastį.

## 4. Laikas pagal procesų skaičių

Pagreitėjimas skaičiuojamas iš `total` laiko, palyginti su tuo pačiu variantu su 1 procesu (Bendrosios dalies 5 skyrius).

`mpi-base` plečiasi gerai: n = 4000 – 1,95× su 2, 3,83× su 4, 7,15× su 8 ir **8,85× su 12 procesų**. Daugyba tokia lėta (209,7 s su 1 procesu – tiek pat, kiek `seq-base`), kad nuosekli dalis nereikšminga (f < 0,3 %), o perdavimų kaina nepastebima. Tai, kad šiam variantui padeda ir E branduoliai, greičiausiai lemia jo pobūdis: branduolys daugiausia laukia atminties (Bendrosios dalies 4.1 skyrius), todėl P ir E branduolių greičio skirtumas jam mažesnis. Tačiau geras plečiamumas nereiškia greito darbo – žr. 5 skyrių.

![mpi-opt vykdymo laikas pagal procesų skaičių (logaritminė Y ašis): greičiausia 8 procesų konfigūracija (išskyrus n = 2000, kur 4 procesai); su 12 procesų laikas vėl auga, didžiausioms matricoms – beveik dvigubai](reports/ataskaita/img/mpi/plots/mpi-opt-time.png)

![mpi-opt pagreitėjimas: iki 4 procesų beveik idealus, su 8 – 5,6–6,2× vidutinėms matricoms, o 12 procesų visada lėtesni už 8; didžiausioms matricoms (punktyras) 8 ir 12 procesų rezultatai krenta](reports/ataskaita/img/mpi/plots/mpi-opt-speedup.png)

| n | 2 procesai | 4 procesai | 8 procesai | 12 procesų |
|---|-----------|-----------|-----------|-----------|
| 2000 | 1,76× | 2,64× | 2,57× | 1,88× |
| 4000 | 1,89× | 3,45× | **5,56×** | 4,84× |
| 8000 | 2,00× | 3,79× | **6,15×** | 5,72× |
| 12000 | 2,04× | 3,87× | 5,70× | 2,85× |
| 14000 | 2,11× | 3,93× | 4,84× | 2,62× |
| 16000 | 2,04× | 3,88× | **4,64×** | **2,32×** |

Pastebėjimai:

1. **Iki 4 procesų – beveik idealu.** Dideliems n pagreitėjimas 3,8–3,9× su 4 procesais; su 2 procesais jis net šiek tiek viršija 2× (2,04–2,11×) – galimai todėl, kad du procesai naudoja abiejų P branduolių grupių L2 podėlius (po 16 MB).
2. **Mažos matricos – fiksuotos sąnaudos.** Kai n = 2000, visa programa su 4 procesais trunka 0,085 s; transliavimas, surinkimas ir sinchronizacija nemažėja su procesų skaičiumi, todėl 8 procesai nebepagreitina.
3. **Didžiausias pagreitėjimas – vidutinėms matricoms:** 6,2× su 8 procesais, kai n = 8000. Didesnėms jis krenta (4,6× ties n = 16000) – tai atminties stygius (3.1 skyrius), ne skaičiavimas.
4. **12 procesų visada lėtesni už 8.** Eilutės padalytos po lygiai, bet 4 iš 12 procesų tenka taupiesiems (E) branduoliams, kurie šiame skaičiavimo branduolyje atlieka tik ≈ 0,15 našiojo darbo (Bendrosios dalies 5 skyrius). Visi procesai `MPI_Gatherv` laukia lėčiausiojo: profiliuojant n = 4000 vieno proceso skaičiavimo laikas su 12 procesų svyravo 0,168–0,219 s, su 8 – tik 0,207–0,216 s. Didelėms matricoms prisideda ir 12 B kopijų (iki 12,3 GB). **Apkrovos balansavimas** (svertinis padalijimas arba dinaminis darbų skirstymas) sąmoningai nerealizuotas – efektas paliktas kaip nevienodų branduolių demonstracija.

## 5. Laikas pagal matricų dydį

![Vykdymo laikas pagal matricos dydį, kiekvienam n – geriausias procesų skaičius (abi ašys logaritminės): mpi-opt kreivė lygiagreti n³ nuo n ≈ 4000 iki 12000 ir stačiau kyla toliau, mpi-base šuolis ties n ≈ 3600](reports/ataskaita/img/mpi/plots/size.png)

![Tas pats laikas pagal operacijų skaičių n·k·p (geriausias procesų skaičius): kol vienos operacijos kaina pastovi, taškai guli tiesėje; mpi-opt tiesė lūžta ties n = 12000](reports/ataskaita/img/mpi/plots/size-linear.png)

| n | mpi-base, 1 p. | mpi-base, 8 p. | mpi-base, 12 p. | mpi-opt, 1 p. | mpi-opt, 8 p. | mpi-opt, 12 p. | mpi-base / mpi-opt, 8 p. |
|---|---------|---------|---------|--------|--------|--------|--------|
| 2000 | 4,74 s | 0,874 s | 0,871 s | 0,223 s | 0,087 s | 0,119 s | 10× |
| 4000 | 209,7 s | 29,3 s | 23,7 s | 1,66 s | **0,300 s** | 0,344 s | **98×** |
| 8000 | – | – | – | 13,1 s | 2,13 s | 2,30 s | |
| 16000 | – | – | – | 106,3 s | **22,9 s** | 45,8 s | |

(„p.“ – procesai.)

- **`mpi-base` atsisakyta ties n = 4000.** Tarp n = 3500 ir 3750 jo laikas su 1 procesu šokteli nuo 49 iki 167 s – tai TLB riba n ≈ 3547 (Bendrosios dalies 4.1 skyrius). Procesų skaičius jos nepanaikina: kiekvienas procesas vis tiek skaito visą B stulpeliais. Su 1 procesu `mpi-opt` greitesnis **126 kartus**, su 8 – 98, su 12 – 69 kartus.
- **`mpi-opt` su 1 procesu laikosi n³ dėsnio:** padvigubinus n (8000 → 16000) laikas išauga 8,1 karto (idealiai 2³ = 8), vienai operacijai tenka pastovios ≈ 0,026 ns. Su 8 procesais tas pats padvigubinimas laiką padidina **10,7 karto**, su 12 – **19,9 karto**: daugiau procesų ir didesnės matricos reiškia daugiau B kopijų ir didesnį atminties stygių.
- Šalia n³ daugybos B pildymas (n²) ir transliavimas tampa vis mažesne laiko dalimi – tai Gustafsono dėsnis (6 skyrius).

## 6. Amdalo dėsnis ir Karpo–Flato metrika

Nuosekli dalis išmatuojama tiesiogiai: f = fill2 / total `mpi-opt` paleidime su 1 procesu (Bendrosios dalies 5 skyrius). Kadangi `mpi-opt` B transliavimą įskaito į daugybą, `fill2` yra grynas nuoseklus pildymas. `mpi-base` taip neanalizuojamas – jo `fill2` apima `MPI_Bcast`.

| n | f | S(8) | S(8) pagal Amdalą | e(8) | S(12) | S(12) pagal Amdalą | e(12) |
|---|---|------|-------------------|------|-------|--------------------|-------|
| 2000 | 6,01 % | 2,57× | 5,63× | 30,1 % | 1,88× | 7,22× | 49,0 % |
| 4000 | 3,20 % | 5,56× | 6,54× | 6,3 % | 4,84× | 8,88× | 13,5 % |
| 8000 | 1,62 % | 6,15× | 7,18× | 4,3 % | 5,72× | 10,18× | 10,0 % |
| 12000 | 1,09 % | 5,70× | 7,43× | 5,8 % | 2,85× | 10,71× | 29,2 % |
| 14000 | 0,92 % | 4,84× | 7,52× | 9,3 % | 2,62× | 10,90× | 32,6 % |
| 16000 | 0,85 % | 4,64× | 7,55× | **10,4 %** | 2,32× | 10,98× | **37,9 %** |

1. **Gustafsono dėsnis.** f mažėja nuo 6,0 % (n = 2000) iki 0,85 % (n = 16000), o riba 1/f kyla nuo 17 iki 118. Užduoties reikalaujamas nuoseklus B pildymas didelėms matricoms beveik neturi įtakos.
2. **Iki 4 procesų viską paaiškina Amdalo dėsnis.** Kai n ≥ 6000, e(4) = 0,6–2,2 %, t. y. lygi išmatuotai f (0,85–2,1 %): kitų nuostolių nėra.
3. **Kreivė U formos.** Mažiausioms matricoms (n ≤ 2500) e didelė dėl fiksuotų MPI sąnaudų (transliavimas, surinkimas, sinchronizacija), kai visa programa trunka vos 0,1–0,4 s. Vidutinėms (n = 6000–12000) e(8) = 4–6 % – nedidelis, beveik pastovus nuostolis. Didžiausioms e vėl auga.
4. **Nuostolis, augantis su procesų skaičiumi.** Kai n = 16000, e = 1,0 % su 4, **10 % su 8 ir 38 % su 12 procesų**, nors f vos 0,85 %. Toks e augimas reiškia, kad kiekvienas papildomas procesas ne tik nepadeda, bet ir lėtina kitus – būtent taip turėtų veikti atminties stygius, kai kiekvienas procesas prideda dar vieną B kopiją (3.1 skyrius).
5. **Nevienodi branduoliai to nepaaiškina.** Įvertinus E branduolių silpnumą (N_eff = 8,6 vietoj 12, Bendrosios dalies 5 skyrius), Amdalo dėsnis n = 16000 prognozuoja ≈ 8,1×, o išmatuota 2,3×. E branduoliai paaiškina, kodėl 12 procesų nepagreitina, bet ne tai, kad jie *lėtina*.

![mpi-opt išmatuotas pagreitėjimas (ištisinės linijos) ir Amdalo dėsnio prognozė su išmatuota f (punktyras): iki 4 procesų didelėms matricoms jos sutampa, su 8 ir 12 procesų išmatuotos kreivės lieka gerokai žemiau](reports/ataskaita/img/mpi_amdahl/plots/mpi-opt-amdahl.png)

![Karpo–Flato metrika e: su 4 procesais, kai n ≥ 6000, e beveik sutampa su išmatuota f, su 8 ir 12 procesų e didėja tiek mažiausioms (fiksuotos sąnaudos), tiek didžiausioms matricoms (atminties stygius)](reports/ataskaita/img/mpi_amdahl/plots/mpi-opt-karpflatt.png)

![mpi-opt fazių laikas: B pildymas (raudona) – maža laiko dalis visiems dydžiams; su 12 procesų didžiausioms matricoms auga daugyba, ne pildymas](reports/ataskaita/img/mpi_amdahl/plots/mpi-opt-phases.png)

## Išvados

1. **MPI užduoties reikalavimus atitinka natūraliai:** A eilutes kiekvienas procesas pildo pats, B pildoma rango 0 procese ir transliuojama, daugyba padalijama eilutėmis, rezultatas surenkamas rango 0 procese. Kurso šabloną reikėjo pataisyti – jame procesai daugino skirtingus duomenis, o pranešimų žymės dideliems n viršijo `MPI_TAG_UB`.
2. **Greitį pirmiausia lemia skaičiavimo branduolys, ne pranešimai.** Perėjus nuo i-j-z ciklo prie `gemm.hpp`, `mpi-opt` su 8 procesais greitesnis 98 kartus (n = 4000: 29,3 s → 0,300 s). Vienas `MPI_Gatherv` vietoj n pranešimų ir konvejeris svarbūs tik tada, kai skaičiavimas jau greitas; viename kompiuteryje konvejeris net lėtina.
3. **Geriausias rezultatas – 8 procesai, po vieną kiekvienam našiajam branduoliui:** 6,2× (n = 8000). 12 procesų visada lėtesni už 8, nes lygus eilučių padalijimas priverčia visus laukti procesų taupiuosiuose branduoliuose.
4. **Atskira atmintis turi kainą viename kompiuteryje:** kiekvienas procesas laiko savo B kopiją (n = 16000 – 8,2 GB su 8 procesais). Atsiradus atminties stygiui, macOS glaudina atmintį ir naudoja keitimo sritį, sistemos laikas sudaro ≈ 10 %, o pagreitėjimas krenta iki 4,6× (8 procesai) ir 2,3× (12 procesų), kai n = 16000. Tas pats paleidimas gali užtrukti 16,3 s arba 23,3 s – priklausomai nuo laisvos atminties.
5. **Amdalo dėsnis paaiškina pagreitėjimą iki 4 procesų**, kai n ≥ 6000 (e ≈ f ≈ 1–2 %). Didesniam procesų skaičiui Karpo–Flato metrika atskleidžia su procesų skaičiumi augantį nuostolį (n = 16000: 10 % su 8 ir 38 % su 12), kurio nuoseklus B pildymas (0,85 %) ir nevienodi branduoliai nepaaiškina.
