# Staigus seq-base sulėtėjimas, kai n > 3500 (TLB perpildymas)

Grafike `size-linear.png` matyti, kad `seq-base` staigiai sulėtėja iškart po
n = k = p = 3500. Šiame dokumente paaiškinama, kodėl taip yra: einant matrix2 stulpeliu
reikia daugiau puslapių, nei telpa į TLB (adresų vertimo buferį). Priežastis – ne duomenų
podėlis.

Kompiuteris: Apple M2 Pro, macOS, puslapio dydis 16 KB (`sysctl hw.pagesize` = 16384).

## 1. Stebėjimas

`seq-base` daugybos laikas (stulpelis `mul`, `results/seq-base-*.csv`),
normuotas pagal daugybos ir sudėties operacijų skaičių (n³):

| n    | mul, s | ns vienai daugybos ir sudėties operacijai |
|------|--------|---------------------|
| 2000 | 4,78   | 0,60                |
| 2500 | 13,6   | 0,87                |
| 3000 | 25,6   | 0,95                |
| 3500 | 49,9   | 1,16                |
| 3550 | 56,2   | 1,26                |
| 3600 | 84,5   | **1,81**            |
| 3700 | 161,2  | **3,18**            |
| 3750 | 168,7  | **3,20**            |
| 4000 | 211,2  | **3,30**            |

Tarp n = 3550 ir n = 3700 vienos operacijos kaina išauga beveik tris kartus. Toliau
ji vėl stabili – apie 3,2–3,3 ns. Tai šuolis, o ne laipsniškas augimas.

## 2. Kreipimosi į atmintį šablonas

`seq-base` naudoja i-j-z ciklų tvarką. Vidinis ciklas eina matrix2 **stulpeliu**:

```cpp
for (int i = 0; i < a.n; ++i)
    for (int j = 0; j < a.p; ++j) {
        int sum = 0;
        for (int z = 0; z < a.k; ++z) sum += m1.at(i, z) * m2.at(z, j);
        m3.at(i, j) = sum;
    }
```

Matricos saugomos eilutėmis (row-major), todėl gretimus stulpelio elementus skiria
`4·p` baitų: 14 000 baitų, kai p = 3500, ir 16 000 baitų, kai p = 4000. Puslapiai yra
16 KB, todėl beveik kiekvienas `z` žingsnis patenka į **kitą puslapį**. Vienas ėjimas
stulpeliu paliečia apie

```
pages per column ≈ 4 · k · p / 16384
```

puslapių. Praktiškai tai yra visa matrix2.

Kitas stulpelis (`j + 1`) yra tuose pačiuose puslapiuose. Jo adresų vertimus galima
panaudoti pakartotinai tik tuo atveju, jei visi jie dar tebėra TLB (translation
lookaside buffer).

## 3. Kodėl riba yra ties ~3550

Apple M1/M2 našiųjų (P) branduolių L2 TLB, remiantis trečiųjų šalių matavimais, turi
apie **3072 įrašus** (Apple šio skaičiaus neskelbia). Esant 16 KB puslapiams, tai
apima 48 MB atminties.

Staigus sulėtėjimas prasideda ten, kur puslapių skaičius vienam stulpeliui viršija
TLB talpą:

```
4 · k · p / 16384 > 3072   =>   k = p > sqrt(3072 · 16384 / 4) ≈ 3547
```

Tai sutampa su išmatuota riba tarp n = 3500 ir n = 3600.

- **Žemiau ribos:** visi stulpelio adresų vertimai lieka TLB ir yra panaudojami
  pakartotinai kitam stulpeliui.
- **Virš ribos:** ciklinis ėjimas stulpeliais pašalina kiekvieną įrašą anksčiau, nei jo
  vėl prireikia, – tai blogiausias atvejis LRU tipo politikai. Tada beveik kiekvienas
  kreipinys nepataiko į TLB ir reikalauja **puslapių lentelės peržiūros**, kuri
  kainuoja kelis papildomus kreipinius į atmintį. Perėjimas išsiplečia per intervalą
  3550–3700, nes TLB yra asociatyvus rinkiniais (set-associative) ir ne griežtai LRU.
  Po to beveik kiekvienas kreipinys nepataiko, o kaina išlieka pastovi.

## 4. Eksperimentas: riba priklauso nuo matrix2 dydžio, o ne nuo n

Norint atskirti TLB poveikį nuo darbo kiekio, n buvo užfiksuotas ties 200, o keičiamas
tik k = p (matrix2 dydis):

```sh
for s in 3400 3500 3550 3580 3600 3650 3700 4000; do
    build/seq-base -n 200 -k $s -p $s --seed 1 --csv
done
```

| k = p | puslapių stulpeliui | ns vienai daugybos ir sudėties operacijai |
|-------|------------------|---------------------|
| 3400  | 2822             | 1,03                |
| 3500  | 2990             | 1,20                |
| 3550  | 3076 (≈ riba)    | 1,36                |
| 3580  | 3129             | 1,72                |
| 3600  | 3164             | 1,88                |
| 3650  | 3252             | 2,67                |
| 3700  | 3342             | 3,19                |
| 4000  | 3906             | 3,28                |

Tas pats šuolis pasirodo ir esant n = 200, kai visas paleidimas trunka kelias
sekundes, ir lygiai ties ta pačia puslapių stulpeliui reikšme. Tai atmeta
paaiškinimus, pagrįstus bendru darbo kiekiu ar šiluminiu dažnio ribojimu (thermal
throttling).

L2 duomenų podėlis taip pat nėra priežastis. Vieno stulpelio podėlio eilutės užima
apie n × 128 B ≈ 460 KB, kai n = 3600, ir lengvai telpa į 16 MB L2. Ribojantis
išteklius yra skirtingų **puslapių** skaičius, o ne baitų skaičius.

## 5. Kuriems variantams tai turi įtakos

| variantas                                   | ciklų tvarka | kreipimasis į matrix2  | paveiktas |
|---------------------------------------------|------------|------------------------|----------|
| seq-base, openmp-base, mpi-base             | i-j-z      | stulpeliais (puslapio dydžio žingsnis) | taip      |
| openmp-opt, openmp-opt2                     | i-k-j      | eilutėmis (nuosekliai) | ne       |

Variantuose `opt` ir `opt2` viename 16 KB puslapyje telpa 4096 iš eilės einantys
eilutės elementai, todėl TLB įrašas panaudojamas 4096 kartus iš eilės. Kai n = 4000,
visi trys i-j-z variantai su viena gija (procesu) užtrunka apie 212 s. `opt` užtrunka
4,9 s, o `opt2` – 1,7 s.

## 6. Pasekmės matavimams

- Kai n ≥ 3600, bet koks pagreitėjimas, apskaičiuotas `seq-base` atžvilgiu, yra
  stipriai išpūstas. Pavyzdžiui, kai n = 4000, opt2 su viena gija yra apie 120×
  greitesnis už `seq-base`. Didžioji to dalis kyla iš kreipimosi į atmintį šablono,
  o ne iš lygiagretumo. Dėl šios priežasties pagreitėjimo grafikai
  (`<variant>-speedup.png`) lygina kiekvieną variantą su jo paties laiku, gautu su
  1 gija (procesu).
- Lyginant dydžius, reikia apsiriboti dydžiais žemiau ribos (šioje mašinoje
  n ≤ 3500) arba aiškiai pažymėti sritį virš jos.
- Virš ribos i-j-z variantai vis tiek gerai skaliuojasi didinant gijų skaičių (apie
  7× su 8 gijomis, kai n = 4000). Kiekvienas branduolys turi savo TLB, todėl gijos dėl
  jo nekonkuruoja. Jos tiesiog visos moka tą pačią baudą už kiekvieną kreipinį.

## 7. Galimi sprendimai (netaikyti, baziniai variantai įgyvendina užduotį taip, kaip ji pateikta)

- i-k-j ciklų tvarka arba blokavimas (skaidymas blokais), kaip `opt` / `opt2`.
- matrix2 transponavimas prieš daugybą, kad vidinis ciklas abu operandus skaitytų
  eilutėmis.
- Didesni puslapiai pakeltų ribą, tačiau macOS neleidžia naudotojui valdyti puslapio
  dydžio įprastiems atminties išskyrimams.
