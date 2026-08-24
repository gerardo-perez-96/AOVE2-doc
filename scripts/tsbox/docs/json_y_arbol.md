# Dos cosas de esta sesión: un bug de pérdida de datos y el árbol de derivadas

## El bug: "eliminaba la columna" era permanente y con mensaje falso

Descripción del usuario: al recargar, el JSON "guardaba que había eliminado
la columna". Diagnóstico exacto:

`_prune_missing_columns()` comparaba las columnas del proyecto contra
`self.df.columns` — el DataFrame **ya filtrado** por lo que se pidió cargar
*en esa sesión concreta*. Si abrías pidiendo menos columnas que la vez
anterior (o cambiabas de `long_mode`), el resto se interpretaba como "ya no
existe en el fichero" y se borraba **del proyecto**, con un mensaje que
mentía: *"la columna ya no está en el fichero"* cuando sí estaba.

Reproducido:

```
1ª apertura, todas las columnas  -> 18 series, guarda JSON
2ª apertura, pidiendo solo 2     -> "se elimina" las otras 16, guarda JSON
3ª apertura, todas otra vez      -> siguen sin volver: BORRADO PERMANENTE
```

**Arreglo, en dos partes:**

1. La comprobación de "¿existe la columna?" ahora se hace contra la
   **cabecera real del fichero** (`peek_columns`, 5 filas), no contra la vista
   filtrada de esta apertura. Solo se borra una serie si su columna
   desapareció de verdad del CSV/parquet.
2. Una columna que existe en el fichero pero no se marcó para cargar esta vez
   se oculta (`visible=False`, con `hidden_reason="not_loaded"` persistido en
   el propio `SeriesDef`), no se destruye. Al volver a marcarla, reaparece con
   su historial de anotaciones intacto. El aviso ahora dice la verdad: *"no se
   han cargado esta vez (no estaban marcadas)"*.

Y `values()` ya no revienta con `KeyError` si alguien pide los datos de una
serie no cargada en la sesión actual: devuelve NaN y el panel puede decirlo,
en vez de tirar la aplicación.

Cinco tests de regresión en `tests/test_json_column_bug.py`, incluido el
escenario exacto de las tres aperturas.

## El árbol: derivadas con panel propio, plegado

Pedido: ver las derivadas por separado, no apiladas con eje Y secundario en
el mismo panel que la original — "con lo fea que es la interfaz cuesta verla".

Antes, la única forma de sacar una media móvil o una derivada de su propio
panel era `overlay_on_parent=True`, que la superponía en el **mismo** gráfico
con un eje Y a la derecha. Es lo que hacía difícil de leer: dos señales de
escalas distintas compartiendo espacio.

**Ahora:** la lista de series de la izquierda es un árbol. Cada serie de
origen es una raíz; sus derivadas cuelgan debajo, **plegadas por defecto**.
Cada derivada tiene su **panel propio**, con su propia escala Y automática
(la que se ajustó en la sesión anterior), separado de la original.

- Plegar el nodo oculta los paneles de sus derivadas sin tocar sus checkboxes.
- Expandir los muestra tal como estaban.
- El checkbox manda por encima del plegado: si una derivada está desmarcada,
  expandir su padre no la hace aparecer.
- `overlay=True` sigue existiendo y sigue siendo útil — pinta la derivada
  superpuesta en el panel del padre **además de** tener su panel propio
  plegado. Son dos cosas independientes, no alternativas: una es para ver la
  relación con el original de un vistazo, la otra para analizarla a fondo con
  su propia escala.

Ocho tests en `tests/test_derived_tree.py`: panel propio, plegado por
defecto, expandir/plegar, el checkbox mandando, varias derivadas del mismo
padre, borrado en cascada, series sin hijos.
