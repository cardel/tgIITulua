# CLAUDE.md

Guía para sesiones de Claude Code en este repositorio.

## Qué es este repositorio

Extracción y categorización de los trabajos de grado de la Universidad del Valle
(2012–2018), con un clasificador basado en modelos de lenguaje bajo
`llm-classifier/`.

---

## Navegación de código: graft

Este repositorio está indexado por [graft](https://github.com/NanoNets/Graft),
que construye con tree-sitter un grafo de símbolos y lo expone por MCP. Para
orientarse o navegar código se consulta **antes** que `grep`, `sed`, `Read` de
archivos completos o el agente `Explore`: una consulta suele reemplazar varias
lecturas y cuesta una fracción de los tokens.

| Pregunta | Tool |
|---|---|
| ¿Cómo funciona X? ¿Dónde está Y? | `graft_find_code` |
| Necesito **todas** las ocurrencias | `graft_find_all` |
| ¿Quién llama a esto? ¿Qué rompo si lo cambio? | `graft_trace_calls` |
| ¿Qué expone este archivo? | `graft_file_api` |
| Repositorio desconocido, ¿por dónde empiezo? | `graft_repo_map` |

Si los tools llegan diferidos —solo los nombres, sin esquema— se cargan los
cinco de una vez, antes de la primera navegación:

```
ToolSearch "select:mcp__graft__graft_find_code,mcp__graft__graft_find_all,mcp__graft__graft_trace_calls,mcp__graft__graft_file_api,mcp__graft__graft_repo_map"
```

Al cablearlo, el 2026-08-18, el grafo traía 11 tarjetas: los scripts de
extracción y los del clasificador. Los JSON de resultados y los textos
extraídos son datos, no código, y quedan fuera.

### En una máquina nueva

`.mcp.json` viaja en el clon y ahí queda registrado el servidor; el grafo no
viaja, porque `graft/` es caché local y está en `.gitignore`. Si la carpeta no
existe, o si los tools no devuelven nada, se reconstruye una vez desde la raíz
del repositorio:

```bash
npx -y @nanonets/graft@0.10.1 build
```

Es tree-sitter puro: local, sin red y sin mandar una línea de código a ningún
modelo. Pide Node 20 o superior y tarda segundos. La versión va fijada en
`.mcp.json`; conviene construir con esa misma.

Los hooks que rehacen el grafo tras un `pull` o un cambio de rama viven en
`.git/hooks/`, que tampoco viaja en el clon. Para reponerlos:

```bash
for h in post-merge post-checkout; do
  printf '#!/usr/bin/env sh\ncommand -v npx >/dev/null 2>&1 || exit 0\nnpx -y @nanonets/graft@0.10.1 build >/dev/null 2>&1 &\nexit 0\n' > ".git/hooks/$h"
  chmod +x ".git/hooks/$h"
done
```

Con `cardel/claude-work` clonado, `bash scripts/bootstrap-graft.sh` hace las dos
cosas en todos los repositorios de `~/repositorios` de una sola pasada.

---

## Orquestación por niveles

Regla del workspace del 24 de septiembre de 2026. La sesión, que aquí corre con
Opus, orquesta y decide con Carlos; cada subtarea baja al nivel más barato que
la resuelve: `tarea-local` (el modelo de Ollama, manejado por Haiku),
`tarea-basica` (Haiku), `tarea-media` (Sonnet) y, para sacar del hilo una
subtarea que exige juicio, `tarea-compleja` (Opus). Un subagente sin modelo
hereda el de la sesión, así que `Explore` y `general-purpose` no se usan a
secas. Contrato de cada subtarea: rutas exactas, tajada de 30 KB o menos,
archivo de salida y respuesta de diez líneas; commits, rastreadores y
decisiones no se delegan. Los cuatro agentes están enlazados en `.claude/agents/`
de este proyecto; la tabla completa está en `~/repositorios/work/CLAUDE.md`.
