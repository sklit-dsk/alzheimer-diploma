# Analiza značaja karakteristika za skup podataka ImagesOASIS

Ovaj dokument prikazuje osnovnu analizu skupa podataka **ImagesOASIS / OASIS Alzheimer's Detection**, koji je planiran za upotrebu u diplomskom radu na temu **„Informacioni sistem za rano otkrivanje Alchajmerove bolesti“**. Skup podataka je preuzet sa Kaggle platforme: https://www.kaggle.com/datasets/ninadaithal/imagesoasis. Podaci potiču iz OASIS projekta, koji se koristi u istraživanjima starenja mozga i Alchajmerove bolesti: https://sites.wustl.edu/oasisbrains/.

## Opis skupa podataka

Skup podataka sadrži MRI slike mozga u JPG formatu. Ukupno je dostupno **86.437 slika**, raspoređenih u četiri klase:

| Klasa | Broj slika | Broj jedinstvenih subjekata |
|---|---:|---:|
| Non Demented | 67.222 | 266 |
| Very mild Dementia | 13.725 | 58 |
| Mild Dementia | 5.002 | 21 |
| Moderate Dementia | 488 | 2 |

Najveći dio podataka pripada klasi **Non Demented**, dok je klasa **Moderate Dementia** veoma slabo zastupljena. Zbog toga je za početnu analizu značaja karakteristika uzet scenario koji je najbliži temi ranog otkrivanja bolesti: poređenje klasa **Non Demented** i **Very mild Dementia**.

Ovakav izbor je važan jer se diplomski rad ne fokusira samo na prepoznavanje uznapredovalih stadijuma bolesti, već prije svega na mogućnost ranog razlikovanja osoba bez demencije od osoba kod kojih se već pojavljuju vrlo blagi znaci promjena.

## Način podjele podataka

Kod rada sa MRI slikama postoji rizik da više sličnih presjeka istog pacijenta završi i u trening i u test skupu. To može dovesti do nerealno dobrih rezultata, jer model tada ne uči samo opšte obrasce bolesti, već može prepoznati sličnosti između slika istog subjekta.

Zbog toga podaci nijesu dijeljeni nasumično po slikama, već po identifikatoru subjekta. Na taj način su slike istog subjekta ostale unutar istog dijela skupa podataka.

| Skup | Non Demented | Very mild Dementia | Ukupno |
|---|---:|---:|---:|
| Trening skup | 10.919 | 10.919 | 21.838 |
| Test skup | 2.806 | 2.806 | 5.612 |

Za ovu fazu analize korišćen je balansiran skup, kako veći broj slika iz klase **Non Demented** ne bi previše uticao na rezultate.

## Karakteristike koje su posmatrane

Pošto se radi o slikama, značaj karakteristika nije posmatran kroz klasične kolone kao kod tabelarnih podataka, već kroz mjerljive osobine izvedene iz MRI presjeka. Posmatrane su sljedeće grupe karakteristika:

- opšta svjetlina i kontrast slike,
- odnos tamnih i svijetlih piksela,
- tekstura slike,
- ivice i strukturne granice,
- procijenjena površina regiona mozga,
- lokalne karakteristike po regionima slike.

Slika je podijeljena na mrežu regiona, kako bi se moglo vidjeti da li pojedini djelovi MRI presjeka imaju veći značaj za razlikovanje klasa. Ovaj pristup ne zamjenjuje medicinsku interpretaciju snimaka, ali može pomoći da se bolje razumije na koje vizuelne obrasce se model najviše oslanja.

## Rezultati modela

Za procjenu značaja karakteristika korišćen je model Random Forest, jer omogućava relativno jasnu interpretaciju uticaja pojedinačnih karakteristika. Glavni cilj nije bio da se dobije konačni medicinski klasifikator, već da se utvrdi koje osobine slike najviše doprinose razlikovanju klasa **Non Demented** i **Very mild Dementia**.

| Metrika | Vrijednost |
|---|---:|
| Accuracy | 0.609 |
| Balanced accuracy | 0.609 |
| F1 macro | 0.583 |
| ROC AUC | 0.770 |

Dobijeni rezultati pokazuju da model u određenoj mjeri razlikuje dvije klase, ali i da zadatak nije jednostavan. To je očekivano, jer se radi o ranom stadijumu bolesti, gdje vizuelne razlike na MRI presjecima mogu biti suptilne.

Ove metrike ne treba posmatrati kao konačnu ocjenu budućeg sistema. One služe kao početna procjena i kao osnova za objašnjenje značaja pojedinih karakteristika. U kasnijoj fazi rada ima smisla koristiti konvolucione neuronske mreže i dodatne metode objašnjivosti, kao što su Grad-CAM ili slične tehnike.

## Najznačajnije karakteristike

U tabeli su prikazane karakteristike koje su imale najveći uticaj na rezultat modela. Značaj je mjeren metodom permutation importance, gdje se posmatra koliko se performanse modela pogoršavaju kada se vrijednosti određene karakteristike poremete.

| Rang | Karakteristika | Tip karakteristike | Značaj |
|---:|---|---|---:|
| 1 | `dark_ratio` | raspodjela intenziteta | 0.0163 |
| 2 | `global_p10` | globalna svjetlina i kontrast | 0.0063 |
| 3 | `edge_mean` | ivice i strukturne granice | 0.0056 |
| 4 | `roi_r1_c1_std` | lokalni intenzitet i kontrast | 0.0048 |
| 5 | `roi_r2_c4_std` | lokalni intenzitet i kontrast | 0.0043 |
| 6 | `roi_r2_c2_std` | lokalni intenzitet i kontrast | 0.0030 |
| 7 | `roi_r1_c1_mean` | lokalni intenzitet i kontrast | 0.0029 |
| 8 | `global_mean` | globalna svjetlina i kontrast | 0.0028 |
| 9 | `roi_r1_c3_std` | lokalni intenzitet i kontrast | 0.0024 |
| 10 | `roi_r3_c3_mean` | lokalni intenzitet i kontrast | 0.0024 |

Najveći značaj imala je karakteristika `dark_ratio`, koja opisuje udio tamnijih piksela u slici. Pored nje, među važnim karakteristikama se nalaze i globalni intenzitet slike, prosječna izraženost ivica, kao i lokalne karakteristike pojedinih regiona slike.

Ovo ukazuje da model ne koristi samo jednu opštu osobinu slike, već kombinuje informacije o intenzitetu, kontrastu, teksturi i lokalnim promjenama u različitim djelovima MRI presjeka.

## Značaj po grupama karakteristika

| Grupa karakteristika | Ukupan pozitivni značaj |
|---|---:|
| Lokalni intenzitet i kontrast | 0.0407 |
| Raspodjela intenziteta | 0.0166 |
| Globalna svjetlina i kontrast | 0.0102 |
| Ivice i strukturne granice | 0.0057 |
| Tekstura | 0.0025 |
| Geometrija regiona mozga | 0.0025 |

Najveći zbirni značaj imaju lokalne karakteristike intenziteta i kontrasta. To je korisno za dalji rad, jer pokazuje da regionalne informacije iz MRI slike mogu biti važnije od same prosječne svjetline cijele slike.

## Tumačenje za diplomski rad

Rezultati ove analize mogu se koristiti kao početni dokaz da MRI slike sadrže mjerljive karakteristike koje pomažu u razlikovanju osoba bez demencije od osoba sa vrlo blagim znacima demencije. Posebno je važno to što su se kao značajne pokazale lokalne karakteristike slike, a ne samo globalni pokazatelji.

U okviru diplomskog rada ovaj dio se može predstaviti kao analiza objašnjivosti nad izvedenim karakteristikama MRI slika. Kasnije, kada se bude razvijao glavni model zasnovan na neuronskim mrežama, ovaj pristup se može dopuniti metodama koje prikazuju važne regione slike direktno na MRI presjeku.

## Ograničenja analize

Postoji nekoliko ograničenja koja treba navesti u radu. Prvo, slike su dostupne kao 2D JPG presjeci, a ne kao originalni 3D medicinski formati kao što su DICOM ili NIfTI. Zbog toga dio informacija iz originalnog snimanja može biti izgubljen.

Drugo, skup podataka nije ravnomjerno raspoređen po klasama. Klasa **Moderate Dementia** ima veoma mali broj subjekata, pa se na osnovu nje ne mogu donositi pouzdani zaključci.

Treće, iako je podjela po subjektima bolja od obične nasumične podjele po slikama, za ozbiljnu medicinsku primjenu bila bi potrebna dodatna validacija na nezavisnom skupu podataka. Za potrebe diplomskog rada ova analiza može poslužiti kao dobra osnova za objašnjavanje pristupa i za dalji razvoj modela.

## Zaključak

Analiza pokazuje da se kod ranog razlikovanja klasa **Non Demented** i **Very mild Dementia** najviše ističu karakteristike vezane za raspodjelu intenziteta, lokalni kontrast i ivice na MRI slici. To potvrđuje da se u daljem radu treba fokusirati ne samo na klasifikaciju, već i na objašnjivost modela, kako bi sistem bio razumljiviji i korisniji u kontekstu ranog otkrivanja Alchajmerove bolesti.
