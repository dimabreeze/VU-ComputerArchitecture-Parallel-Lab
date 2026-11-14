# Amdalo dėsnis: teorija ir mūsų matavimai

Grafikai: [`reports/07-amdahl/`](../reports/07-amdahl/README.md), sugeneruoti
[`scripts/plot.py`](../scripts/plot.py) (`--plots amdahl phases fraction karpflatt`,
`--e-ratio`).

## 1. Teorija

### Amdalo dėsnis

Programa turi dalį, kuri vykdoma tik nuosekliai (dalis **f** viso laiko, kai dirba 1 gija ar
procesas), ir dalį, kurią galima padalyti N gijų (procesų) (dalis 1 − f). Idealiu atveju
lygiagrečioji dalis tampa N kartų greitesnė, o nuoseklioji nesikeičia:

```
T(N) = T(1) · (f + (1 − f) / N)
S(N) = T(1) / T(N) = 1 / (f + (1 − f) / N)
```

Net su be galo daug gijų (procesų) pagreitėjimas yra ribotas: **S(∞) = 1 / f**.
Kai f = 5 %, joks branduolių skaičius neduoda daugiau nei 20×.

Dėsnis yra *viršutinė riba*: jis daro prielaidą, kad lygiagrečioji dalis plečiasi idealiai
(vienodi branduoliai, jokios komunikacijos, jokios konkurencijos dėl bendrų išteklių). Kiekvienas toks nuostolis
nuleidžia išmatuotą pagreitėjimą žemiau kreivės.

### Gustafsono dėsnis

Amdalo dėsnyje uždavinio dydis fiksuotas. Gustafsono pastebėjimas: kai uždavinys didėja,
nuoseklioji dalis dažnai auga lėčiau nei lygiagrečioji, todėl f mažėja, o
riba 1/f kyla. Lygiagretieji kompiuteriai labiau atsiperka didesniems uždaviniams.

### Karpo–Flato metrika

Turint išmatuotą pagreitėjimą S su N gijų (procesų), nuoseklioji dalis, kuri pagal Amdalo dėsnį
jį *paaiškintų*, yra

```
e(N) = (1/S − 1/N) / (1 − 1/N)
```

- e ≈ f: išmatuota nuoseklioji dalis visiškai paaiškina pagreitėjimą;
- e > f: yra kitų nuostolių (sinchronizacija, atmintis, lėtesni branduoliai);
- e auga su N: nuostoliai auga didėjant gijų (procesų) skaičiui.

### Nevienodi (heterogeniniai) branduoliai

M2 Pro turi 8 našiuosius (P) ir 4 taupiuosius (E) branduolius. Amdalo dėsnis branduolius laiko
vienodais. Jei E ir P branduolių greičių santykis yra r ir pirmosios 8 gijos (procesai)
veikia P branduoliuose, N gijų (procesų) atitinka

```
N_eff = min(N, 8) + r · max(N − 8, 0)
```

P branduolių, ir formulėje N pakeičiamas N_eff.

## 2. Kaip matuojame f

Užduotis reikalauja matricą B pildyti **nuosekliai**; matricos A pildymas ir
daugyba yra lygiagretūs. Kiekvienas paleidimas fazių laikus pateikia atskirai, todėl nuoseklioji
dalis matuojama tiesiogiai, o ne parenkama pritaikant kreivę:

```
f = fill2 / total     (paleidimas su 1 gija / procesu, tas pats variantas ir dydis)
```

`fill2` auga kaip n² (viena atsitiktinė reikšmė kiekvienam elementui), daugyba – kaip n³, todėl
f ≈ c / n: ji mažėja didėjant matricoms.

`mpi-base` taip neanalizuojamas: jo `fill2` apima B transliavimą `MPI_Bcast`.

### E branduolio greitis r

Išmatuota su `openmp-opt2`, n = 4000, viena gija, daugybos laikas, po 3 paleidimus:

| branduolys | komanda | `mul` |
|------------|---------|-------|
| P | `build/openmp-opt2 -s 4000 -t 1` | 1,60 s |
| E | `taskpolicy -b build/openmp-opt2 -s 4000 -t 1` | 10,85 s |

`taskpolicy -b` paleidžia procesą su foniniu prioritetu, o tokius procesus macOS planuoja
E branduoliuose. r = 1,60 / 10,85 ≈ **0,15**. Tai gali būti per mažas E branduolio įvertis esant įprastam
prioritetui: foninis prioritetas taip pat gali sumažinti taktinį dažnį, o 4 gijos su
`taskpolicy -b` svyravo nuo 3,3 iki 5,2 s. Antrasis įvertis iš esamų `openmp-opt2` paleidimų
(dinaminis paskirstymas, todėl mul(8) / mul(12) = (8 + 4r) / 8) duoda
r = 0,03…0,15, priklausomai nuo n. Abu rodo tą patį: šiame skaičiavimo branduolyje 4 E branduoliai prideda
maždaug tiek, kiek pusė P branduolio. `reports.sh` naudoja r = 0,15.

## 3. Rezultatai

### openmp-opt2

| n     | f      | 1/f  | S(8) išmatuotas | S(8) pagal Amdalą | S(12) išmatuotas | S(12) pagal Amdalą, N_eff = 8,6 |
|-------|--------|------|-----------------|-------------------|------------------|---------------------------------|
| 2000  | 6,00 % | 17   | 5,24            | 5,63              | 5,21             | 5,91                            |
| 4000  | 3,13 % | 32   | 6,18            | 6,56              | 6,38             | 6,94                            |
| 8000  | 1,55 % | 65   | 6,54            | 7,21              | 6,62             | 7,69                            |
| 16000 | 0,84 % | 119  | 6,34            | 7,55              | 6,79             | 8,08                            |

1. **Mažos matricos: Amdalo dėsnis paaiškina didžiąją dalį.** Kai n = 2000, 6 % nuoseklaus
   darbo apriboja 8 gijų pagreitėjimą iki 5,6×; išmatuojame 5,2×.
2. **Didelės matricos: Amdalo dėsnis paaiškina mažai.** Kai n = 16000, nuoseklioji dalis yra
   mažesnė nei 1 % ir dėsnis leidžia 7,5× su 8 gijomis, tačiau išmatuotas pagreitėjimas lieka
   apie 6,3–6,5× visiems n ≥ 4000.
3. **Karpo–Flato metrika parodo likusią dalį.** Su 8 gijomis e ≈ 3–4 % visiems n ≥ 4000, o
   išmatuota f mažėja nuo 3,1 % iki 0,84 %. Papildomas maždaug 3 % nuostolis nemažėja
   didėjant uždaviniui, todėl tai ne nuoseklus pildymas. Tikėtinos priežastys (atskirai nematuotos):
   mažesnis P branduolių taktinis dažnis, kai visi jie užimti, bendras L2 podėlis
   ir atminties pralaidumas, paskutinio bloko laukimas lygiagretaus ciklo pabaigoje.
4. **12 gijų: E branduoliai.** Klasikinė kreivė prognozuoja 7–11× su 12 gijų; išmatuota
   5,2–6,8×, o e šokteli iki maždaug 7 %. Su N_eff = 8,6 kreivė po 8 gijų tampa
   beveik plokščia – tokios pat formos kaip matavimai.
5. **Gustafsonas.** f mažėja nuo 6 % iki 0,84 %, o riba 1/f kyla nuo 17 iki 119;
   todėl `openmp-opt2` pagreitėjimas auga su n (5,2× → 6,3–6,8×).

### Kiti variantai

- `openmp-base`: f < 0,3 %, todėl Amdalo dėsnis leidžia beveik idealų plečiamumą, ir variantas iš tikrųjų
  gerai plečiasi (8,9× su 12 gijų, kai n = 4000) – daugyba tokia lėta, kad
  nuoseklioji dalis neturi reikšmės.
- `openmp-opt`: f yra 0,6–2 %, bet pagreitėjimas mažėja didėjant n (5,9× → 3,9× su 8 gijomis):
  e išauga gerokai virš f. Tai ne Amdalo dėsnis, o atminties magistralė (04 etapas).
- `mpi-opt`: f tokia pati kaip `openmp-opt2` (pildymas tas pats), ir iki
  n = 12000 variantas plečiasi panašiai (5,6–6,2× su 8 procesais). Nuo n = 14000 pagreitėjimas
  krenta: kai n = 16000, jis yra 4,6× su 8 ir 2,3× su 12 procesų, e pasiekia 10 % ir 38 %: tai nuostolis, augantis didėjant procesų skaičiui, greičiausiai
  dėl atminties stygiaus (memory pressure), kurį sukelia kiekvieno proceso B kopijos (16000² int = po 1 GB).
  Toliau netirta.

## 4. Išvados

- Amdalo dėsnis su išmatuota f yra gera viršutinė riba ir paaiškina mažų uždavinių
  pagreitėjimą.
- Dideliems uždaviniams nuoseklioji dalis nereikšminga; tada ribą lemia
  bendri aparatiniai ištekliai ir nevienodi branduoliai, kurių paprastas dėsnis
  nemodeliuoja. Karpo–Flato metrika šiuos du veiksnius atskiria.
- Užduotis neleidžia lygiagretinti B pildymo; didelėms matricoms tai ir taip
  duotų mažai (mažiau nei 1 %, kai n = 16000).
