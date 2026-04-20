# Wie unsere KI lernt — ELI10

Stell dir vor, du sollst ein Buch schreiben, das immer das nächste Wort vorhersagt. Wenn jemand "Der Hund rennt über die..." liest, soll dein Buch "Straße" vorschlagen. Je besser du rätst, desto niedriger dein Score. Wir wollen den niedrigsten Score der Welt.

Das Problem: Dein Buch darf nur **16 Megabyte** groß sein (kleiner als ein Handyfoto) und du hast nur **10 Minuten** zum Lernen — allerdings auf 8 der schnellsten Computer-Chips der Welt.

---

## 1. Das Gehirn: Transformer

Unser Modell ist wie eine Kette von Filtern. Text geht rein, und jeder Filter versteht den Text ein bisschen besser.

**Schicht 1** erkennt: "Das ist ein Nomen."
**Schicht 5** erkennt: "Das ist ein Tier, das etwas tut."
**Schicht 10** erkennt: "Nach 'über die' kommt wahrscheinlich ein Ort."

Wir haben **11 solcher Filter** (genannt "Layer"). Jeder Filter hat zwei Teile:

- **Attention** ("Aufmerksamkeit"): Schaut sich alle bisherigen Wörter an und entscheidet, welche wichtig sind. Wie wenn du in einem Buch zurückblätterst, um den Kontext zu verstehen.
- **MLP** ("Verarbeitung"): Nimmt die gesammelten Infos und berechnet daraus neue Features. Wie wenn du das Gelesene in eigene Worte fasst.

---

## 2. Attention: Wer ist wichtig?

Stell dir vor, du liest den Satz "Die Katze, die gestern im Garten war, frisst..." — um das nächste Wort zu raten, musst du wissen, dass es um eine Katze geht, nicht um den Garten.

So funktioniert Attention:

1. Jedes Wort stellt eine **Frage** (Query): "Wer ist für mich relevant?"
2. Jedes Wort bietet einen **Schlüssel** (Key): "Das bin ich, das kann ich."
3. Jedes Wort hat einen **Wert** (Value): "Das ist meine Information."

Die Frage wird mit allen Schlüsseln verglichen. Je besser ein Schlüssel zur Frage passt, desto mehr zählt der dazugehörige Wert.

Wir haben **8 parallele Aufmerksamkeitsköpfe** — wie 8 Leute, die den gleichen Text lesen, aber auf verschiedene Dinge achten. Einer achtet auf Grammatik, einer auf das Thema, einer auf die Stimmung.

---

## 3. Positionsmarkierung: RoPE

Das Modell weiß von sich aus nicht, ob ein Wort an Position 1 oder Position 100 steht. RoPE löst das, indem es die Wort-Vektoren **dreht** — wie die Zeiger einer Uhr. Position 1 dreht ein bisschen, Position 100 dreht viel. So kann das Modell erkennen: "Dieses Wort war 5 Positionen vor mir" vs. "50 Positionen vor mir."

---

## 4. Unsere Tricks (was uns besser macht als andere)

### Trick 1: Layer Looping — mehr Tiefe ohne mehr Gewicht

Normalerweise hat jeder der 11 Filter eigene Gewichte. Wir lassen die **Filter 3, 4 und 5 dreimal hintereinander laufen** — mit den gleichen Gewichten. So hat unser Modell effektiv 17 Filter, bezahlt aber nur für 11.

Wie wenn du ein Sieb dreimal benutzt: beim ersten Mal fängt es die großen Stücke, beim zweiten die mittleren, beim dritten die kleinen — obwohl es das gleiche Sieb ist.

Wir schalten das erst nach 35% des Trainings ein, weil es am Anfang verwirrend für das Modell wäre.

### Trick 2: QK-Gain — schärferer Fokus

Normal verteilt sich die Aufmerksamkeit relativ gleichmäßig über alle Wörter. Wir drehen den "Kontrast" hoch (Faktor 5.5 statt normal ~3.3). Das heißt: wenn ein Wort wichtig ist, bekommt es fast die gesamte Aufmerksamkeit. Wie wenn du beim Lesen einen Textmarker benutzt statt alles gleich hell zu lesen.

Jeder der 8 Köpfe lernt seinen eigenen Kontrast-Wert.

### Trick 3: Parallel Residuals — schnellere Verarbeitung

In den oberen Schichten (ab Layer 7) lassen wir Attention und MLP **gleichzeitig** arbeiten statt nacheinander. Beide lesen den gleichen Input und ihre Ergebnisse werden addiert.

Wie zwei Schüler, die unabhängig die gleiche Aufgabe lösen und dann ihre Antworten kombinieren.

### Trick 4: U-Net Skip Connections — Abkürzungen

Die frühen Schichten erkennen einfache Muster (Wortarten, Satzzeichen). Die späten Schichten brauchen diese Info manchmal direkt. Statt sich auf den normalen Weg zu verlassen, bauen wir **Abkürzungen**: Schicht 2 schickt ihre Ergebnisse direkt an Schicht 9.

Wie Querverweise in einem Buch: "Siehe Kapitel 2."

### Trick 5: XSA — keine Wiederholungen

Wenn Schicht 5 schon erkannt hat "das ist eine Katze", muss Schicht 6 das nicht nochmal lernen. XSA entfernt aus jedem Layer die Information, die schon im Value-Vektor steckt. So wird jede Schicht gezwungen, **etwas Neues** beizutragen.

---

## 5. Der Lehrer: Muon Optimizer

Das Modell lernt durch Ausprobieren. Es rät ein Wort, schaut wie falsch es lag, und passt seine Gewichte an. Die Frage ist: **wie** passt es die Gewichte an?

Normale Methoden (wie Adam) behandeln manche Gewichte als wichtiger als andere — Gewichte, die sich oft ändern, werden vorsichtiger angepasst.

**Muon** macht etwas ganz anderes: Es nimmt den Gradienten (= "in welche Richtung soll ich die Gewichte ändern") und **orthogonalisiert** ihn. Das bedeutet: alle Richtungen werden gleich stark verändert, keine Richtung dominiert.

Stell dir vor, du stehst auf einem Hügel und willst runter. Adam geht vorsichtig in Richtungen, wo es steil ist, und schneller wo es flach ist. Muon ignoriert die Steilheit und geht in alle Richtungen gleich schnell — aber in der richtigen Gesamtrichtung.

Verschiedene Teile des Modells lernen mit verschiedenen Geschwindigkeiten:
- **Wörter-Tabelle** (Embedding): langsam und vorsichtig
- **Filter-Gewichte** (Matrizen): schnell und mutig
- **Feintuning-Knöpfe** (Skalare): mittel

---

## 6. EMA: Der Durchschnitt gewinnt

Während des Trainings schwanken die Gewichte hin und her — mal etwas zu weit in eine Richtung, mal zu weit in die andere. Statt die letzten Gewichte zu nehmen, berechnen wir einen **gleitenden Durchschnitt** über das gesamte Training.

Wie wenn du beim Bogenschießen statt dem letzten Schuss den Durchschnitt aller Schüsse nimmst — der ist meist näher am Zentrum.

---

## 7. Schrumpfen: GPTQ Quantisierung

Unser trainiertes Modell ist zu groß. Jedes Gewicht ist eine Dezimalzahl mit 32 Bit Genauigkeit (z.B. 0.0234567). Wir brauchen so viel Genauigkeit nicht.

**GPTQ** reduziert jedes Gewicht auf nur **6 Bit** (64 mögliche Stufen statt 4 Milliarden). Das ist wie wenn du von HD-Fotos auf Pixel-Art umsteigst — du verlierst Details, aber das Bild ist noch erkennbar.

Der clevere Teil: Wenn beim Runden eines Gewichts ein Fehler entsteht, verteilt GPTQ diesen Fehler auf die noch nicht gerundeten Gewichte, sodass sie den Fehler ausgleichen. Dafür benutzt es die **Hessian-Matrix** — die sagt, welche Gewichte besonders wichtig sind und genauer gerundet werden müssen.

- Filter-Gewichte: 6 Bit (stark komprimiert)
- Wörter-Tabelle: 8 Bit (etwas vorsichtiger)
- Kleine Einstellknöpfe: 16 Bit (kaum komprimiert, sind eh winzig)

---

## 8. Noch kleiner: Byte Shuffle + Brotli

Nach der Quantisierung sortieren wir die Bytes um: Alle ersten Bytes zusammen, alle zweiten zusammen. Ähnliche Werte landen nebeneinander, was die Kompression verbessert.

Dann kommt **Brotli** (ein Kompressionsalgorithmus wie ZIP, nur besser). Das Endergebnis ist unter 16 MB.

---

## 9. Test-Time Training: Lernen bei der Prüfung

Das ist unser geheimster Trick. Normalerweise wird ein Modell trainiert und dann benotet — fertig. Wir lassen das Modell **während der Prüfung weiterlernen**.

So funktioniert es:

1. Das Modell bekommt einen Textblock (32.000 Tokens)
2. **Erst benoten**: Es rät alle Wörter und wir schreiben die Note auf (ohne zu schummeln!)
3. **Dann lernen**: Das Modell trainiert auf diesem Textblock weiter
4. Der nächste Textblock profitiert davon, weil das Modell sich an den Schreibstil angepasst hat

Das ist legal, weil:
- Jedes Wort wird benotet, **bevor** das Modell darauf trainiert
- Kein Wort wird zweimal benotet
- Es wird nicht geschummelt (keine Manipulation der Vorhersagen)

Wie wenn du bei einer Prüfung die ersten Aufgaben löst und dabei merkst: "Aha, dieser Lehrer mag Scherzfragen." Die späteren Aufgaben löst du besser, weil du den Stil verstanden hast — aber die Noten der ersten Aufgaben ändern sich nicht.

---

## 10. Das große Bild

```
TRAINING (10 Minuten):

8 Milliarden Wörter Text
        ↓
  [Transformer mit 11 Layern]
  [+ Layer Looping = 17 virtuelle Layer]
  [+ Parallel Residuals ab Layer 7]
  [+ QK-Gain 5.5 für scharfen Fokus]
  [+ Skip Connections wie U-Net]
        ↓
  Muon Optimizer (gleich stark in alle Richtungen)
        ↓
  EMA (Durchschnitt aller Trainingsstände)
        ↓
  GPTQ (32 Bit → 6 Bit, clever gerundet)
        ↓
  Brotli Kompression (< 16 MB)


PRÜFUNG:

  Lade komprimiertes Modell
        ↓
  Für jeden Textblock:
    1. Benoten (Note zählt)
    2. Weiterlernen (für nächsten Block)
        ↓
  Finale Note: X.XX Bits pro Byte
  (aktueller Weltrekord: 1.0810)
```

Unser Ziel: unter 1.0810 kommen. Jede Nachkommastelle zählt.
