"""
SRBIN Nikola Tesla, za sva vremena, najveci naucnik sveta.

SERBIAN Nikola Tesla, for all time, the greatest scientist in the world.
"""



"""
Tesla_k-wave-python_2A.py  —  GRUPA 2, varijanta 2A (k-wave-python motor)

Ista struktura kao GRUPA 1, samo jaci motor:
  motor (k-wave FDTD talasno polje)  ->  primena na 4630 izvlacenja  ->  skor  ->  rangirane kombinacije
Umesto rucnog d'Alambera (grupa 1), talas racuna k-wave-python (pseudospektralni FDTD).

Iz polja uzimamo liniju u pravcu prostiranja:
  S(x)   = pritisak duz pravca prostiranja (skalarno polje)
  E_x    = -dS/dx  (uzduzna komponenta, analog SLW)

Deljene funkcije (ucitavanje CSV, skor, frekvencija, kombinacije, crtanje)
uvozim iz grupe 1 da struktura ostane identicna.
"""

from pathlib import Path

import numpy as np

from Tesla_Scalar_1 import (
    SEED,
    W_TALAS,
    W_FREQ,
    CSV_PATH,
    MIN_BROJ,
    MAX_BROJ,
    OUTPUT_DIR,
    ucitaj_izvlacenja,
    glavne_mere,
    ne_frekvencijski_skor,
    frekvencija_brojeva,
    kombinovani_skor,
    izaberi_kombinacije,
    skor_kombinacije,
    nacrtaj_polje,
)

OSNOVA = "tesla_k-wave-python_2A"


def simuliraj_kwave(nx=4630, sirina_grida=256, visina_grida=64, dx=0.1e-3):
    """k-wave FDTD motor. Vrati (x, S, E_x) duz pravca prostiranja, duzine nx.

    Postavi Gaussov pritisni puls levo, pusti ga da se prostire u +x,
    pa iz finalnog polja uzmi liniju kroz sredinu (pravac prostiranja).
    """
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksource import kSource
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    Nx, Ny = sirina_grida, visina_grida
    kgrid = kWaveGrid([Nx, Ny], [dx, dx])
    medium = kWaveMedium(sound_speed=1500, density=1000)
    kgrid.makeTime(medium.sound_speed)

    # Pocetni Gaussov puls (po x) blizu leve ivice, ravan po y.
    xx = np.arange(Nx)
    centar, sirina = Nx * 0.2, Nx * 0.03
    profil = np.exp(-((xx - centar) ** 2) / (2.0 * sirina ** 2))
    p0 = np.tile(profil[:, None], (1, Ny))

    source = kSource()
    source.p0 = p0

    sensor = kSensor(mask=np.ones((Nx, Ny), dtype=bool), record=["p_final"])

    rezultat = kspaceFirstOrder(
        kgrid, medium, source, sensor,
        pml_inside=False, smooth_p0=False, quiet=True,
    )

    p_final = np.asarray(rezultat["p_final"]).reshape(Nx, Ny)
    linija = p_final[:, Ny // 2]  # pravac prostiranja (+x)

    # Interpoliraj liniju na nx tacaka (= broj izvlacenja).
    x = np.linspace(0.0, (Nx - 1) * dx, nx)
    S = np.interp(np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, Nx), linija)
    E_x = -np.gradient(S, x[1] - x[0])
    return x, S, E_x


def main():
    # --- Korak 1: motor (k-wave) ---
    izvlacenja = ucitaj_izvlacenja()
    n = len(izvlacenja)
    x, S, E_x = simuliraj_kwave(nx=n)
    mere = glavne_mere(S, E_x)
    print()
    print("Tesla Scalar / GRUPA 2 - 2A (k-wave-python motor)")
    print("Talas: akusticko polje (FDTD), linija u pravcu prostiranja")
    print("Uzduzno polje: E_x = -dS/dx")
    print()
    print(f"broj tacaka: {len(x)}")
    print(f"max S: {mere['max_S']:.10f}")
    print(f"max |E_x|: {mere['max_abs_E_x']:.10f}")
    print(f"ukupna gustina energije: {mere['ukupna_gustina_energije']:.10f}")
    print()

    # --- Korak 2: primena talasa na CSV + prava frekvencija ---
    energija = 0.5 * (S ** 2 + E_x ** 2)
    talas_skor, _ = ne_frekvencijski_skor(izvlacenja, energija)
    udeo, pojave = frekvencija_brojeva(izvlacenja)
    skor = kombinovani_skor(talas_skor, udeo)
    poredak = sorted(skor.items(), key=lambda kv: kv[1], reverse=True)
    freq_poredak = sorted(pojave, key=lambda b: (pojave[b], b), reverse=True)
    kombinacije = izaberi_kombinacije(skor, broj_kombinacija=10, seed=SEED)
    rangirane_kombinacije = sorted(
        ((k, skor_kombinacije(k, skor)) for k in kombinacije),
        key=lambda kv: kv[1],
        reverse=True,
    )
    png, jpg = nacrtaj_polje(x, S, E_x, osnova=OSNOVA)

    with open(OUTPUT_DIR / f"{OSNOVA}.txt", "w", encoding="utf-8") as f:
        f.write("Tesla Scalar - GRUPA 2 / 2A (k-wave-python: talas + prava frekvencija)\n")
        f.write(f"CSV: {CSV_PATH}\n")
        f.write(f"Izvlacenja: {n} | Seed: {SEED} | tezine: talas={W_TALAS} freq={W_FREQ}\n\n")
        f.write("Brojevi po kombinovanom skoru (tezinski talas + frekvencija):\n")
        for b, s in poredak:
            f.write(f"  {b:02d}  skor={s:.10f}  freq={udeo[b]:.5f}  (pojava={pojave[b]})\n")

        f.write("\nTabela pravih frekvencija (opadajuce po freq, pa po broju):\n")
        f.write("  broj | pojava |   udeo\n")
        f.write("  -----+--------+--------\n")
        for b in freq_poredak:
            f.write(f"   {b:02d}  |  {pojave[b]:4d}  | {udeo[b]:.5f}\n")
        f.write(f"  ukupno pojava: {sum(pojave.values())}\n")

        f.write("\nPredlozene kombinacije (rangirane po skoru kombinacije):\n")
        for i, (k, s_komb) in enumerate(rangirane_kombinacije, start=1):
            f.write(f"  {i:02d}. " + " ".join(f"{v:02d}" for v in k) + f"  skor_komb={s_komb:.10f}\n")

        f.write("\nSlike talasa/polja:\n")
        f.write(f"  PNG: {png}\n")
        f.write(f"  JPG: {jpg}\n")

    print()
    print("\nTesla Scalar - GRUPA 2 / 2A (k-wave-python: talas + prava frekvencija)")
    print(f"CSV: {CSV_PATH} | Izvlacenja: {n} | tezine: talas={W_TALAS} freq={W_FREQ}")
    print("\nTop 10 brojeva po kombinovanom skoru (tezinski talas + frekvencija):")
    for b, s in poredak[:10]:
        print(f"  {b:02d}  skor={s:.10f}  freq={udeo[b]:.5f}  (pojava={pojave[b]})")

    print()
    print("\nTabela pravih frekvencija (opadajuce po freq, pa po broju):")
    print("  broj | pojava |   udeo")
    print("  -----+--------+--------")
    for b in freq_poredak:
        print(f"   {b:02d}  |  {pojave[b]:4d}  | {udeo[b]:.5f}")
    print(f"  ukupno pojava: {sum(pojave.values())}")

    print()
    print("\nPredlozene kombinacije (rangirane po skoru kombinacije):")
    for i, (k, s_komb) in enumerate(rangirane_kombinacije, start=1):
        print(f"  {i:02d}. " + " ".join(f"{v:02d}" for v in k) + f"  skor_komb={s_komb:.10f}")
    print(f"\nSacuvano: {OUTPUT_DIR / f'{OSNOVA}.txt'}")
    print()


if __name__ == "__main__":
    main()



"""
Tesla Scalar - GRUPA 2 / 2A (k-wave-python: talas + prava frekvencija)
CSV: /Users/4c/Desktop/GHQ/data/loto7hh_4630_k46.csv | Izvlacenja: 4630 | tezine: talas=0.7 freq=0.3

Top 10 brojeva po kombinovanom skoru (tezinski talas + frekvencija):
  34  skor=0.9121841284  freq=0.02694  (pojava=873)
  14  skor=0.7895833333  freq=0.02496  (pojava=809)
  08  skor=0.6983287873  freq=0.02808  (pojava=910)
  24  skor=0.6931807556  freq=0.02592  (pojava=840)
  33  skor=0.6791768738  freq=0.02635  (pojava=854)
  16  skor=0.6572597948  freq=0.02583  (pojava=837)
  19  skor=0.6508913033  freq=0.02508  (pojava=813)
  06  skor=0.6504542415  freq=0.02518  (pojava=816)
  11  skor=0.5945810001  freq=0.02654  (pojava=860)
  01  skor=0.5662804330  freq=0.02431  (pojava=788)


Tabela pravih frekvencija (opadajuce po freq, pa po broju):
  broj | pojava |   udeo
  -----+--------+--------
   08  |   910  | 0.02808
   23  |   905  | 0.02792
   34  |   873  | 0.02694
   26  |   869  | 0.02681
   37  |   860  | 0.02654
   11  |   860  | 0.02654
   32  |   857  | 0.02644
   33  |   854  | 0.02635
   22  |   851  | 0.02626
   39  |   849  | 0.02620
   29  |   848  | 0.02616
   10  |   845  | 0.02607
   35  |   843  | 0.02601
   09  |   843  | 0.02601
   38  |   842  | 0.02598
   07  |   842  | 0.02598
   24  |   840  | 0.02592
   25  |   839  | 0.02589
   16  |   837  | 0.02583
   31  |   830  | 0.02561
   13  |   828  | 0.02555
   05  |   828  | 0.02555
   21  |   826  | 0.02549
   03  |   825  | 0.02546
   02  |   824  | 0.02542
   28  |   820  | 0.02530
   18  |   820  | 0.02530
   06  |   816  | 0.02518
   19  |   813  | 0.02508
   04  |   812  | 0.02505
   12  |   810  | 0.02499
   14  |   809  | 0.02496
   15  |   797  | 0.02459
   27  |   788  | 0.02431
   01  |   788  | 0.02431
   30  |   787  | 0.02428
   36  |   786  | 0.02425
   20  |   770  | 0.02376
   17  |   766  | 0.02363
  ukupno pojava: 32410


Predlozene kombinacije (rangirane po skoru kombinacije):
  01. 05 10 11 23 33 34 39  skor_komb=4.1437050539
  02. 01 08 13 19 22 30 34  skor_komb=4.1424132679
  03. 16 24 25 27 29 31 33  skor_komb=4.0744426473
  04. 01 09 14 19 23 33 38  skor_komb=4.0198240763
  05. 04 06 11 13 14 27 39  skor_komb=3.8875155185
  06. 04 06 09 11 29 30 39  skor_komb=3.5942499308
  07. 08 09 19 27 28 31 37  skor_komb=3.5795413950
  08. 06 09 12 20 22 24 33  skor_komb=3.4345352628
  09. 07 15 16 21 25 32 33  skor_komb=3.1977739458
  10. 04 09 23 27 31 36 37  skor_komb=3.1048084795

Sacuvano: /Users/4c/Desktop/GHQ/Tesla/tesla_k-wave-python_2A.txt

Slike talasa/polja:
  PNG: /Users/4c/Desktop/GHQ/Tesla/tesla_k-wave-python_2A.png
  JPG: /Users/4c/Desktop/GHQ/Tesla/tesla_k-wave-python_2A.jpg
"""



"""
Motor: simuliraj_kwave koristi k-wave-python (kspaceFirstOrder, backend='python') — Gaussov pritisni puls kreće s leve strane, uzimam liniju u pravcu prostiranja → S(x), pa E_x = -dS/dx. Interpolira se na 4630 tačaka.
Ostalo isto kao grupa 1: skor, prava frekvencija, kombinovani skor (0.7/0.3), rangirane kombinacije, crtež. Deljene funkcije uvozim iz Tesla_Scalar_1 da struktura bude identična.
Imena izlaza: tesla_k-wave-python_2A.txt, .png, .jpg.

treba instaliran k-wave-python (pip install k-wave-python)  

što je najvažnije, k-wave motor daje drugačiji talas od grupe 1, ne kopiju.

Top broj je opet 34, ali odmah ispod je 14 (skor 0.79) — a 14 je po frekvenciji slab (809, pri dnu tabele). 
Znači talas ga je podigao, ne frekvencija. To je tačno ono što tražim: ne-frekvencijski signal iz prave FDTD simulacije.
Slično 19 i 06 su u top 10 iako su frekvencijski ispod proseka → opet talas radi.
Kombinacije su potpuno drugačije od grupe 1.
Favorit 2A: 05 10 11 23 33 34 39 (skor_komb 4.1437).
Poređenje grupa 1 vs 2A je smisleno: 
ista struktura, drugi motor, drugačiji rezultat 
"""



"""
Postoji teorijska osnova (EED/SLW) + alati za simulaciju talasa. 


GRUPA 2 

2. Gotove biblioteke za simulaciju talasa/EM polja 


2A k-wave-python — simulacija talasnih polja (akustika, FDTD)

2B pycharge — EM polja/potencijali pokretnih naboja (JAX, GPU)

2C Wakis — 3D EM solver (računa i longitudinalne komponente)
           (najbliže Teslinom SLW)

2D rfx — diferencijabilni 3D FDTD EM simulator
         (može učenje/optimizacija)

k-wave-python — prvo. Najbliže grupi 1 (FDTD talasno polje).
pycharge — drugo. Uvodi prava polja naboja (JAX/GPU), dobra provera da li „izvor" menja rezultat.
Wakis — treće. Pravi 3D EM solver sa longitudinalnim komponentama → ovo je srce Tesline SLW priče.
rfx — poslednje. Diferencijabilni FDTD → kad sve radi, njime optimizujemo parametre (učenje težina, ne ručno 0.7/0.3).

Logika: 
prve dve daju temelj i poređenje, treća donosi pravi longitudinalni talas, četvrta pretvara ceo sistem u nešto što se može podešavati/učiti.

Svaka varijanta = ista struktura kao grupa 1 (motor → primena na 4630 → skor → rangirane kombinacije), samo jači motor.
"""



"""
Analiza — Tesla 2A (k-wave-python FDTD motor)

Motor: k-wave-python, FDTD/pseudospektralna simulacija talasnog polja. 
Ovo je fizički ozbiljniji talasni motor od Tesla 1. 
Uzimam liniju kroz 2D polje u pravcu prostiranja, pa iz nje S(x) i E_x = -dS/dx.

Top brojevi (talas + freq, 0.7/0.3): 34 (0.912) · 14 (0.790) · 08 (0.698) · 24 (0.693) · 33 (0.679) · 16 (0.657) · 19 (0.651) · 06 (0.650) · 11 (0.595) · 01 (0.566)

34 ostaje #1, kao i u Tesla 1 → stabilan signal kroz dva različita motora.
14 je #2 iako je frekvencijski slab (809, pri dnu) → jak ne-frekvencijski talasni efekat.
19, 06, 01 su takođe nisko/srednje po frekvenciji, ali visoko po skoru → k-wave motor jasno menja rang.
35, koji je bio #2 u Tesla 1, ispada iz top 10 → znači 2A nije samo kopija Tesla 1.
Favorit kombinacija: 05 10 11 23 33 34 39 (skor_komb = 4.1437). 
Druga je skoro izjednačena: 01 08 13 19 22 30 34 (4.1424).

Zaključak: 2A je bolji „fizički" kandidat od Tesla 1 jer koristi pravi talasni solver. 
Daje drugačiju topologiju skora, ali zadržava neke stabilne brojeve (34, 08, 33). 
Posebno je interesantan broj 14, jer ga frekvencija ne objašnjava.
"""



"""
source ~/tesla_env/bin/activate

Bitne verzije za tesla_env:

Paket	Verzija
python  3.11.13
numpy   2.2.6
scipy   1.15.3
pandas  3.0.3
matplotlib    3.10.9
k-Wave-python 0.6.2
pycharge      2.0.1
jax        0.10.1
jaxlib     0.10.1
jaxtyping  0.3.7
equinox    0.13.8
lineax     0.1.1
optimistix 0.1.0
ml-dtypes
(uz jax)
opencv-python 4.13.0.92
h5py          3.16.0
"""
