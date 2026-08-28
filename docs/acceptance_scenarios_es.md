# Escenarios de aceptación en español

Estos escenarios derivan directamente de los ejemplos y requisitos de
[`Task.txt`](Task.txt). Separan dos comprobaciones distintas:

- Las pruebas automatizadas ejercitan el agente, las herramientas y SQLite con
  un límite de modelo programado. Son deterministas y no llaman a OpenAI.
- La comprobación manual usa el micrófono y las API reales para validar
  transcripción, interpretación del español y síntesis de voz.

## Pruebas automatizadas

Ejecutar desde la raíz del repositorio:

```powershell
.venv\Scripts\python -m pytest tests\test_acceptance_es.py -q
```

| Escenario | Requisito comprobado |
| --- | --- |
| Gestión completa de la cesta | Añadir, eliminar, cambiar cantidad, consultar y confirmar |
| Pedido de la semana anterior | Recuperar un pedido, quitar leche y añadir aguacates |
| «Necesito café» | Consultar preferencias y frecuencia antes de proponer Lavazza |
| Preferencia de leche de avena | Guardar memoria explícita y usarla en una conversación nueva |
| «Hazme la compra habitual» | Reconstruir la cesta con productos y cantidades frecuentes |
| Historial confirmado | Conservar fecha, productos y cantidades en SQLite |

Estas pruebas no demuestran por sí solas que un modelo concreto interprete
cualquier formulación en español. Demuestran que, cuando el modelo solicita las
acciones esperadas, el bucle de herramientas y el estado persistente producen
el resultado correcto.

## Preparación de la comprobación manual

Configurar las credenciales solo en el entorno; no escribir la clave en ningún
archivo del repositorio:

```powershell
$env:OPENAI_API_KEY = "tu-clave"
$env:OPENAI_LLM_MODEL = "tu-modelo-con-function-calling"
.venv\Scripts\python -m grocery_agent.gradio_app `
  --database data/aceptacion-es-01.sqlite3 `
  --user usuario-aceptacion `
  --port 7861 `
  --open-browser
```

Usar un nombre de base de datos nuevo para repetir la prueba desde un estado
vacío. En cada paso de voz, comprobar que la transcripción conserva la intención
de la frase, aunque no sea literalmente idéntica, y que la respuesta reproducida
está en español.

## Escenario 1: cesta y pedido

1. En `Voz`, decir: «Añade dos litros de leche».
2. Decir: «Añade cuatro tomates y dos yogures».
3. Decir: «Quita los tomates. En vez de dos yogures quiero seis».
4. Preguntar: «¿Qué llevo en la cesta?».
5. Decir: «Confirma el pedido».

Resultado esperado:

- Antes de confirmar, la cesta contiene dos litros de leche y seis yogures.
- Los tomates ya no aparecen.
- Tras confirmar, la cesta queda vacía.
- `Historial de pedidos` muestra la fecha, ambos productos y sus cantidades.

## Escenario 2: compra anterior con cambios

Preparación:

1. Añadir dos litros de leche y un paquete de arroz.
2. Confirmar el pedido y pulsar `Nueva conversación`.

Prueba:

1. Decir: «Necesito hacer la compra para esta semana».
2. Cuando el agente proponga usar el pedido anterior, responder: «Sí, pero esta
   vez no necesito leche y añade cuatro aguacates».

Resultado esperado: la cesta contiene el arroz y cuatro aguacates, pero no
contiene leche. El pedido histórico original no cambia.

## Escenario 3: memoria activa para café

Preparación:

1. Decir: «Recuerda que mi marca de café preferida es Lavazza».
2. Añadir un paquete de café Lavazza, confirmarlo y crear una conversación nueva.

Prueba: decir «Necesito café».

Resultado esperado:

- El estado indica consultas a `get_preferences` y `get_frequent_items`.
- El agente propone el café Lavazza y la cantidad habitual en vez de preguntar
  genéricamente qué café se desea.
- La cesta no cambia hasta responder afirmativamente.

## Escenario 4: preferencia entre conversaciones

1. Decir: «La próxima vez recuerda que prefiero leche de avena».
2. Pulsar `Nueva conversación`.
3. Decir: «Añade dos litros de leche».

Resultado esperado: el agente recuerda la preferencia, propone leche de avena y
espera confirmación. Después de confirmarla, la cesta contiene dos litros de
leche de avena.

## Escenario 5: compra habitual

Preparación:

1. Confirmar un pedido con dos manzanas y un paquete de arroz.
2. Confirmar otro pedido con cuatro manzanas y un paquete de arroz.
3. Crear una conversación nueva.

Prueba: decir «Hazme la compra habitual».

Resultado esperado: el agente consulta `get_frequent_items` y reconstruye, o
propone reconstruir, una cesta con tres manzanas —la cantidad media— y un
paquete de arroz. Si solicita confirmación, aceptarla antes de comprobar la
cesta.

## Criterio global de aceptación

El prototipo supera la comprobación cuando los cinco escenarios mantienen el
estado correcto en los paneles de SQLite, conservan memoria tras `Nueva
conversación`, responden oralmente en español y no modifican la cesta ante una
selección ambigua sin confirmación.
