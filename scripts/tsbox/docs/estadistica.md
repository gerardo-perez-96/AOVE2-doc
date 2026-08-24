# Lo que la ventana de análisis hace y por qué

Esta parte no es UI. Son decisiones que cambian las conclusiones que sacas.
Si las ignoras, la herramienta te dará números bonitos y falsos.

---

## 1. El problema de fondo: los p-valores clásicos no valen aquí

La fórmula estándar de significancia de una correlación asume que tus muestras
son **independientes**. Las series temporales nunca lo son: el valor de ahora se
parece al de hace un segundo.

Analogía: quieres saber la opinión media de una ciudad y preguntas a 1000
personas. Si las 1000 son de la misma familia, no tienes 1000 opiniones, tienes
aproximadamente una. El test estadístico, sin embargo, se cree que tienes 1000 y
te da una precisión que no existe.

Consecuencia concreta, medida en `tests/test_analysis.py`: dos paseos aleatorios
**totalmente independientes** de 1000 puntos salen significativos con p<0.05 en
más de 20 de cada 30 intentos con el test clásico. Es el resultado clásico de
Yule (1926): correlación espuria.

**Qué hace tsbox.** Calcula el *tamaño muestral efectivo* de Bartlett:

    n_eff = n · (1 − ρ₁ˣ·ρ₁ʸ) / (1 + ρ₁ˣ·ρ₁ʸ)

y usa `n_eff` en lugar de `n` para el p-valor. La columna "n efectivo" de la
matriz te enseña el daño: si con 50.000 muestras el n efectivo es 300, tus
series tienen tanta inercia que en la práctica solo has observado 300 cosas
independientes.

Se desactiva con la casilla, pero desactívala sabiendo qué estás perdiendo.

---

## 2. Comparaciones múltiples

Con 20 series haces 190 tests. A α=0.05, esperas ~10 "significativos" aunque
todo sea ruido. Es como tirar 190 monedas y sorprenderte de que alguna saque
diez caras seguidas.

**Qué hace tsbox.** Benjamini-Hochberg sobre el triángulo superior de la matriz.

- **Fondo verde** = pasa FDR. Esto es lo que puedes creerte.
- **Fondo marrón** = solo pasa p<α en crudo. Sospechoso.
- **Fondo azul/rojo** = no significativo; el color es la magnitud y el signo de r.

Mira el verde. El marrón es la lista de cosas que investigar, no de cosas ciertas.

---

## 3. Correlación cruzada: prewhitening

La CCF entre dos señales con inercia sale llena de picos anchos aunque no exista
ninguna relación. Ambas cambian despacio, así que cualquier desfase produce
solape.

Analogía: dos ríos crecidos a la vez. Correlacionan perfectamente. No es que uno
alimente al otro: es que llovió sobre los dos, y la lluvia es lenta.

**Prewhitening** ajusta un AR(p) a X y aplica *ese mismo filtro* a X y a Y. Lo
que queda de X es su parte impredecible — la sorpresa. Si esa sorpresa aparece en
Y unos lags después, ahí sí hay información. Es el procedimiento de Box-Jenkins.

Está activado por defecto. Al desactivarlo, la herramienta te lo dice en la barra
naranja.

**Convenio de signo:** `lag > 0` significa que **X adelanta a Y** (X en `t` se
compara con Y en `t+lag`). O sea: lag positivo ⇒ X es candidato a predictor.

**El p-valor del pico está corregido** por Šidák sobre el número de lags
probados. Si miras 101 lags y te quedas con el mejor, el p sin corregir es
basura: has hecho 101 tests y reportado el ganador.

---

## 4. ACF / PACF

- **ACF** con bandas de **Bartlett**, que se ensanchan con el lag. La banda plana
  ±1.96/√n solo es válida bajo hipótesis de ruido blanco, y si tu serie fuera
  ruido blanco no estarías mirando la ACF.
- **PACF** por Durbin-Levinson. El último lag significativo sugiere el orden AR.
- **Ljung-Box** al pie: si p es minúsculo, la serie tiene memoria y todo p-valor
  de correlación sin corregir que calcules sobre ella miente.
- **Casilla "Diferenciar"**: si la ACF decae lentísimo y casi todo es
  significativo, la serie no es estacionaria. Diferenciar suele arreglarlo, y
  entonces la ACF empieza a decir algo.

Los NaN se interpolan linealmente antes de ACF/CCF (`fill_gaps`) porque estas
funciones exigen muestreo continuo. **Tirar las filas con NaN rompería el
espaciado temporal y desplazaría todos los lags.** Para la correlación puntual
sí se usan casos completos, que ahí no importa el orden.

---

## 5. Histogramas y modos

- Bins por **Freedman-Diaconis**, robusto a colas pesadas. "10 bins" o "√n" te
  esconden modos.
- Los modos se detectan sobre la **KDE**, no sobre el histograma: los picos del
  histograma dependen del binning y te inventan multimodalidad.
- La prominencia mínima es ajustable. Súbela si ves modos fantasma.

**Multimodal casi siempre significa regímenes distintos mezclados** — máquina
encendida/apagada, día/noche, dos sensores en la misma columna. Antes de
modelar, sepáralos: la media de una distribución bimodal es un valor que la
señal casi nunca toma.

---

## 6. Lo que esto NO hace

- **Causalidad.** Un lag significativo dice que X precede a Y, no que lo cause.
  Una tercera variable puede mover a ambas con distinto retraso. Si quieres
  acercarte, mira causalidad de Granger — pero no está aquí.
- **Relaciones no lineales.** Pearson solo ve rectas. Usa Spearman para
  monótonas. Para una U perfecta, ambas dan r≈0 y no hay ninguna independencia.
  El histograma y el gráfico te lo enseñan; el número no.
- **No estacionariedad.** Si la relación entre X e Y cambia a lo largo del
  fichero, la correlación global es un promedio de dos cosas distintas y no
  describe ninguna. Por eso existe la casilla "solo la ventana visible": mide
  por tramos y compara.
