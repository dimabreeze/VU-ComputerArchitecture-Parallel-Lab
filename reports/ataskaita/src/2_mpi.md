title: MPI: procesai su atskira atmintimi
subtitle: mpi-base → mpi-opt
part: 2 dalis

# 2 dalis. MPI

!!! fact "Svarbiausi faktai"
    - Ta pati užduotis, kitas skaičiavimo branduolys: `mpi-opt` už kurso šabloną (`mpi-base`) greitesnis **98 kartus su 8 procesais** (n = 4000: 29,3 s → 0,300 s) (2 ir 5 skyriai).
    - Kiekvienas procesas laiko **visą B kopiją**: n = 14000 tai 8 × 784 MB ≈ 6,3 GB. macOS praneša apie atminties stygių, o tas pats paleidimas trunka **23,3 s arba 16,3 s** – priklausomai nuo laisvos atminties prieš paleidimą (3 skyrius).
    - Pagreitėjimas su 8 procesais didžiausias vidutinėms matricoms – **6,2× (n = 8000)** – ir krenta iki **4,6× (n = 16000)**; 12 procesų visada lėtesni už 8, n = 16000 – tik **2,3×** (4 skyrius).
    - Amdalo dėsnis paaiškina pagreitėjimą iki 4 procesų; didelėms matricoms atsiranda **su procesų skaičiumi augantis nuostolis**, kurio nuoseklus B pildymas nepaaiškina (6 skyrius).

## 1. Kaip užduotis įgyvendinta su MPI

MPI programą sudaro keli **procesai**, kurių kiekvienas turi savo atmintį; duomenimis jie keičiasi tik pranešimais. `mpirun -np 8 build/mpi-opt -s 14000` paleidžia 8 to paties kodo kopijas, o jų vaidmenį skiria **rangas** (0…N−1). Viename kompiuteryje Open MPI pranešimus perduoda per bendrai naudojamą atmintį – tai pačių procesų atliekamas kopijavimas.

1. **A pildoma lygiagrečiai:** kiekvienas procesas pildo savo A eilutes (duomenys nepriklauso nuo procesų skaičiaus, Bendrosios dalies 2.3 skyrius).
2. **B pildoma nuosekliai** rango 0 procese ir **transliuojama** visiems (`MPI_Bcast`; `mpi-opt` – `MPI_Ibcast` + `MPI_Wait`, kurie be konvejerio atlieka tą patį vieną transliavimą).
3. **Daugyba lygiagreti:** kiekvienas procesas skaičiuoja savo C eilutes.
4. **C surenkama** rango 0 procese: `mpi-base` – po vieną eilutę `MPI_Send` / `MPI_Recv`, `mpi-opt` – vienu **surinkimu** `MPI_Gatherv` su `MPI_IN_PLACE`.

Be šių, naudojami tik `MPI_Init`/`MPI_Finalize`, `MPI_Comm_size`/`MPI_Comm_rank`, `MPI_Barrier` (fazių pradžia) ir `MPI_Wtime` (laikas). Fazės matuojamos kaip Bendrosios dalies 2.3 skyriuje; dėl `MPI_Barrier` fazės laikas – lėčiausio proceso laikas.

## 2. Programos: mpi-base ir mpi-opt

**`mpi-base`** (`src/mpi/base.cpp`) – kurso šablonas „MPI program sample“, pataisytas, bet **algoritmas nepakeistas** (cikliškas eilučių paskirstymas, i-j-z ciklas, `MPI_Send` / `MPI_Recv`). Pataisyta:

- duomenys pagal užduotį: šablone kiekvienas procesas generavo *savo* matricas su `rand()`; dabar A pildoma lygiagrečiai, B – rango 0 procese ir transliuojama;
- pranešimų žymė `10 + i` dideliems n viršija `MPI_TAG_UB` – dabar viena žymė srautui (MPI išlaiko tvarką tarp tos pačios procesų poros);
- pašalintas perteklinis `}` prieš `MPI_Finalize`; ataskaitą spausdina tik rangas 0;
- pridėta: argumentai, deterministiniai generatoriai, `--verify`, CSV ataskaita.

```cpp
for (int i = rank; i < a.n; i += size)            // mpi-base: cikliškai, i-j-z
    for (int j = 0; j < a.p; ++j) {
        int sum = 0;
        for (int z = 0; z < a.k; ++z) sum += m1.at(i, z) * m2.at(z, j);
        m3.at(i, j) = sum;                        // vėliau kiekviena eilutė – MPI_Send
    }
```

**`mpi-opt`** (`src/mpi/opt.cpp`) naudoja skaičiavimo branduolį iš `include/gemm.hpp` (tą patį kaip `seq-opt`, Bendrosios dalies 4.3 skyrius) – jis apeina TLB ribą, į kurią atsiremia i-j-z ciklas (Bendrosios dalies 4.1 skyrius). Kiekvienas procesas gauna **ištisinį eilučių bloką** ir saugo tik savo A ir C eilutes; C surenkama vienu kvietimu, rango 0 blokas nekopijuojamas. B transliavimas įskaitomas į `mul`, todėl `fill2` – grynas nuoseklus pildymas.

```cpp
gemm_add(my_rows, k0, k1, j0, j1, m1.data_ptr(), a.k, m2.data_ptr(), a.p,
         m3.data_ptr(), a.p);                     // mpi-opt: savo eilučių blokas
MPI_Gatherv(MPI_IN_PLACE, 0, MPI_INT, m3.data_ptr(), cnt.data(), dsp.data(),
            MPI_INT, 0, MPI_COMM_WORLD);          // rangas 0: vienas surinkimas
```

Konvejeris su `MPI_Ibcast` (B siunčiama dalimis daugybos metu) realizuotas, bet išjungtas: viename kompiuteryje Open MPI neturi asinchroninės pažangos per bendrai naudojamą atmintį, o transliavimas sudaro tik ≈ 6 % daugybos laiko, todėl konvejeris buvo 5–40 % lėtesnis. Išsamiau: `docs/mpi_opt.md`.

## 3. Branduolių apkrova ir atminties stygius

Scenarijus – `mpi-opt`, n = 14000, 8 procesai (stebėjimo būdai – Bendrosios dalies 3 skyrius).

![btop: 8 atskiri procesai po 2 gijas, kiekvienas ≈ 88–95 % branduolio; P branduoliai C4–C11 – 100 %, E branduoliai C0–C3 – 46–67 %](docs/img/screens/mpi-opt-np8-btop.png)
![Šilumos žemėlapis: P branduoliai užimti visą daugybą, E branduoliai – 40–80 % su pliūpsniais iki 100 %; RAM šokteli nuo ≈ 9,7 iki ≈ 13,3 GB, pabaigoje auga swap](docs/img/load/mpi-opt-np8.png)

Kiekvienas MPI procesas procesų sąraše matomas **atskira eilute** (savo PID, 2 gijos). Visi 8 našieji branduoliai užimti 100 %. `/usr/bin/time` duomenimis procesai sunaudojo 101,7 s vartotojo ir **11,8 s sistemos** laiko per 16,3 s – **vidutiniškai 6,96 branduolio**, iš jų ≈ 10 % – sistemos darbas (perdavimas per bendrai naudojamą atmintį, atminties valdymas); vienos gijos `seq-base` – tik ≈ 2 %. Taupieji branduoliai mūsų procesų neturi, bet užimti vidutiniškai ≈ 55 % (prieš paleidimą ≈ 25 %) – tai sistemos darbas, greičiausiai atminties glaudinimas ir keitimo sritis.

!!! fact "Atminties stygius – MPI kaina viename kompiuteryje"
    Procesai neturi bendros atminties, todėl **kiekvienas laiko visą B kopiją**: n = 14000 – po 784 MB, **≈ 6,3 GB 8 procesams** (su A ir C blokais ≈ 9 GB), n = 16000 – po 1 GB. Kitos programos tuo metu jau užima 7–10 GB iš 16 GB. Paleidimo metu macOS atminties stygiaus rodiklis po 3,2 s pakyla nuo 34 % iki **68–71 % (1 lygis – „įspėjimas“)**, auga keitimo sritis, o trukmė priklauso nuo to, kiek atminties laisva:

    | paleidimas | laisva RAM prieš | trukmė | keitimo srities prieaugis | atminties stygius |
    |------------|------------------|--------|---------------------------|-------------------|
    | pirmasis | ≈ 5 GB | **23,3 s** | **+2,1 GB** | iki 71 % |
    | antrasis (žemėlapyje) | ≈ 6,5 GB | **16,3 s** | +0,3 GB | iki 68 % |

Pirmojo paleidimo duomenys užrašyti jo metu, antrojo – `results/monitor/mpi-opt-np8.json(l)`. Trukmė čia – `mpirun` laikas; pačios programos `total` antrajame paleidime – 15,3 s. Naudojama RAM paauga tik ≈ 3,6 GB, o `btop` procesui rodo ≈ 380 MB, nors vien B užima 784 MB; tikėtina, kad macOS suspaudžia B (vien 1 ir −1).

| n | B kopijos: 8 procesai | pagreitėjimas | B kopijos: 12 procesų | pagreitėjimas |
|---|-----------------------|---------------|------------------------|---------------|
| 8000 | 2,0 GB | 6,2× | 3,1 GB | 5,7× |
| 12000 | 4,6 GB | 5,7× | **6,9 GB** | **2,9×** |
| 14000 | **6,3 GB** | **4,8×** | 9,4 GB | 2,6× |
| 16000 | **8,2 GB** | **4,6×** | 12,3 GB | 2,3× |

Pagreitėjimas staigiai krenta, kai visos B kopijos kartu viršija ≈ 6 GB – maždaug tiek, kiek laisvos atminties būna prieš paleidimą. Tai sutapimas, ne įrodymas, bet apkrovos įrašai ir pagreitėjimo kritimas rodo tą pačią priežastį.

## 4. Laikas pagal procesų skaičių

![mpi-opt laikas pagal procesų skaičių (logaritminė ašis): greičiausia – 8 procesai (n = 2000 – 4); su 12 laikas vėl auga](reports/ataskaita/img/mpi/plots/mpi-opt-time.png)
![mpi-opt pagreitėjimas: iki 4 procesų beveik idealus, su 8 – iki 6,2×, 12 procesų visada lėtesni už 8](reports/ataskaita/img/mpi/plots/mpi-opt-speedup.png)

| n | 2 procesai | 4 procesai | 8 procesai | 12 procesų |
|---|-----------|-----------|-----------|-----------|
| 2000 | 1,76× | 2,64× | 2,57× | 1,88× |
| 4000 | 1,89× | 3,45× | **5,56×** | 4,84× |
| 8000 | 2,00× | 3,79× | **6,15×** | 5,72× |
| 16000 | 2,04× | 3,88× | **4,64×** | **2,32×** |

- **Iki 4 procesų – beveik idealu** (3,8–3,9×, kai n ≥ 8000). Mažoms matricoms (n = 2000, 0,085 s) transliavimas, surinkimas ir sinchronizacija nemažėja, todėl 8 procesai nebepadeda.
- **Didžiausias pagreitėjimas – 6,2× (n = 8000, 8 procesai);** didesnėms matricoms jis krenta dėl atminties stygiaus (3 skyrius).
- **12 procesų lėtesni už 8:** eilutės padalytos po lygiai, 4 procesai tenka taupiesiems branduoliams (≈ 0,15 našiojo greičio, Bendrosios dalies 5 skyrius), o visi laukia lėčiausiojo ties `MPI_Gatherv`. Apkrovos balansavimas sąmoningai nepridėtas.
- `mpi-base` plečiasi geriau (n = 4000: 7,2× su 8, 8,9× su 12), nes jo daugyba tokia lėta, kad perdavimai ir nuosekli dalis nereikšmingi, – bet ir su 12 procesų jis trunka 23,7 s.

Išsamiau: `reports/05-mpi/README.md`.

## 5. Laikas pagal matricų dydį

![Laikas pagal matricos dydį, geriausias procesų skaičius (abi ašys logaritminės): mpi-opt lygiagreti n³ nuo n ≈ 4000 iki 12000 ir stačiau kyla toliau; mpi-base šuolis ties n ≈ 3600 (TLB riba)](reports/ataskaita/img/mpi/plots/size.png)

| n | mpi-base, 1 p. | mpi-base, 8 p. | mpi-opt, 1 p. | mpi-opt, 8 p. | mpi-opt, 12 p. |
|---|----------------|----------------|---------------|---------------|----------------|
| 2000 | 4,74 s | 0,874 s | 0,223 s | 0,087 s | 0,119 s |
| 4000 | 209,7 s | 29,3 s | 1,66 s | **0,300 s** | 0,344 s |
| 8000 | – | – | 13,1 s | 2,13 s | 2,30 s |
| 16000 | – | – | 106,3 s | **22,9 s** | 45,8 s |

- **`mpi-base` atsisakyta ties n = 4000:** tarp n = 3500 ir 3750 jo laikas šokteli nuo 49 iki 167 s – TLB riba (Bendrosios dalies 4.1 skyrius), kurios procesų skaičius nepanaikina.
- **`mpi-opt` su 1 procesu laikosi n³:** n padvigubinus (8000 → 16000) laikas išauga 8,1 karto. Su 8 procesais – **10,7 karto**, su 12 – **19,9 karto**: daugiau procesų ir didesnės matricos reiškia daugiau B kopijų.

## 6. Amdalo dėsnis

![mpi-opt išmatuotas pagreitėjimas (ištisinės linijos) ir Amdalo dėsnis su išmatuota f (punktyras): didelėms matricoms iki 4 procesų sutampa, su 8 ir 12 procesų išmatuota gerokai žemiau](reports/ataskaita/img/mpi_amdahl/plots/mpi-opt-amdahl.png)

Nuoseklioji dalis f = fill2 / total (1 procesas) mažėja nuo 6,0 % (n = 2000) iki 0,85 % (n = 16000) – Gustafsono dėsnis, todėl užduoties reikalaujamas nuoseklus B pildymas didelėms matricoms beveik nieko nekainuoja. Kai n ≥ 6000, su 4 procesais Karpo–Flato metrika e lygi f (0,6–2,2 %) – Amdalo dėsnis paaiškina viską. Kai n = 16000, e pasiekia **10 % su 8 ir 38 % su 12 procesų**, nors f vos 0,85 %: nuostolis auga su kiekvienu procesu, kaip ir turėtų būti, kai kiekvienas procesas prideda dar vieną B kopiją. Net įvertinus taupiuosius branduolius (N_eff = 8,6) dėsnis n = 16000 prognozuoja ≈ 8,1×, o išmatuota 2,3×. Išsamiau: `docs/amdahl.md`.

## Išvados

1. **MPI užduotį atitinka natūraliai:** A eilutes pildo kiekvienas procesas, B – rangas 0 ir transliuoja, daugyba dalijama eilutėmis, C surenkama rango 0 procese. Kurso šabloną reikėjo pataisyti: jame procesai daugino skirtingus duomenis, o žymės viršijo `MPI_TAG_UB`.
2. **Greitį lemia skaičiavimo branduolys, ne pranešimai:** `mpi-opt` su 8 procesais 98 kartus greitesnis už `mpi-base` (n = 4000). Vienas `MPI_Gatherv` svarbus tik greitam skaičiavimui, o konvejeris viename kompiuteryje net lėtina.
3. **Geriausia – 8 procesai, po vieną našiajam branduoliui** (6,2×, n = 8000). 12 procesų visada lėtesni: lygus padalijimas verčia visus laukti taupiųjų branduolių.
4. **Atskira atmintis viename kompiuteryje kainuoja:** kiekvienas procesas laiko B kopiją (n = 16000 – 8,2 GB su 8 procesais). Atsiradus atminties stygiui pagreitėjimas krenta iki 4,6× (8 procesai) ir 2,3× (12 procesų), o tas pats paleidimas trunka 16,3 arba 23,3 s – priklausomai nuo laisvos atminties.
5. **Amdalo dėsnis paaiškina pagreitėjimą iki 4 procesų;** toliau Karpo–Flato metrika atskleidžia su procesų skaičiumi augantį nuostolį (n = 16000: 10 % su 8, 38 % su 12), kurio nuoseklus B pildymas (0,85 %) nepaaiškina.
