# PolyTri Algorithmus - Vollständige Implementierungsbeschreibung

## Übersicht

Dieses Dokument beschreibt den vollständigen Algorithmus für Delaunay-Triangulation mit constrained boundaries und hole removal, basierend auf der Python-Implementierung. Die Beschreibung ist sprachunabhängig und enthält alle Details, einschließlich impliziter Operationen.

---

## 1. Datenstrukturen

### 1.1 Grundlegende Strukturen

- **points**: Array von 2D-Punkten (N×2), jeder Punkt ist ein Tupel (x, y) mit Fließkommazahlen
- **triangles**: Liste von Dreiecken, jedes Dreieck ist eine Liste/Array von 3 Punkt-Indizes
- **edge_to_triangles**: Dictionary/Map von Edge-Keys zu Listen von Dreieck-Indizes
- **point_to_triangles**: Dictionary/Map von Punkt-Indizes zu Sets von Dreieck-Indizes
- **boundary_edges**: Set von Boundary-Edges (als Tupel mit Orientierung)

### 1.2 Edge-Normalisierung

**Funktion: `make_edge_key(i1, i2)`**
- **Zweck**: Normalisiert eine Edge zu einem eindeutigen Key, unabhängig von der Reihenfolge
- **Implementierung**:
  ```
  if i1 < i2:
      return (i1, i2)
  else:
      return (i2, i1)
  ```
- **Wichtig**: Edge-Keys werden IMMER normalisiert verwendet für Lookups in `edge_to_triangles`
- **Edge-Tuples**: Behalten die Original-Orientierung (i1, i2) für geometrische Operationen

### 1.3 Index-Mapping

- **point_order**: Array, das die Zuordnung von sortierten Indizes zu originalen Indizes speichert
  - `point_order[i]` = originaler Index des i-ten sortierten Punktes
- **point_unorder**: Array, das die inverse Zuordnung speichert
  - `point_unorder[j]` = sortierter Index des originalen Punktes j
  - Wird berechnet als: `argsort(point_order)`

### 1.4 Konstanten

- **EPS**: Numerisches Epsilon für Fließkomma-Vergleiche = 1.23456789e-14

---

## 2. Initialisierung

### 2.1 Eingabe-Validierung

**Input-Parameter:**
- `points`: Array von 2D-Punkten (mindestens 3 Punkte erforderlich)
- `boundaries`: Optional, Liste von Boundary-Sequenzen (jede Boundary ist eine Liste von Punkt-Indizes)
- `delaunay`: Boolean, ob Delaunay-Kriterium erzwungen werden soll (Standard: true)
- `holes`: Boolean, ob Löcher entfernt werden sollen (Standard: true)
- `border`: Optional, Liste von Boundary-Indizes für hole removal (Standard: leere Liste)

**Validierung:**
1. Prüfe, dass `points` mindestens 3 Punkte enthält
2. Prüfe, dass `points` die Form (N, 2) hat
3. Wenn `boundaries` gegeben:
   - Prüfe, dass jede Boundary mindestens 2 Punkte enthält
   - Prüfe, dass alle Boundary-Indizes gültig sind (0 ≤ index < N)
   - Prüfe, dass keine negativen Indizes vorhanden sind

### 2.2 Initialisierung der Datenstrukturen

```
triangles = []  // Leere Liste
edge_to_triangles = {}  // Leeres Dictionary
point_to_triangles = {}  // Leeres Dictionary
boundary_edges = {}  // Leeres Set
```

---

## 3. Triangulation-Initialisierung

### 3.1 Punkt-Sortierung

**Zweck**: Optimiert die Triangulation durch Sortierung nach Entfernung vom Schwerpunkt

**Algorithmus:**
1. Berechne den Schwerpunkt aller Punkte:
   ```
   center = mean(points, axis=0)
   ```
2. Erstelle Paare von (Punkt, Original-Index):
   ```
   points_with_indices = [(point, original_index) for each point]
   ```
3. Sortiere nach quadrierter Entfernung vom Schwerpunkt:
   ```
   distance_squared(point) = dot(point - center, point - center)
   sort points_with_indices by distance_squared(point)
   ```
4. Reordne Punkte und erstelle Mapping:
   ```
   points = [point for (point, _) in sorted_points]
   point_order = [original_index for (_, original_index) in sorted_points]
   point_unorder = argsort(point_order)  // Inverse Mapping
   ```

**Wichtig**: Die Punkte werden intern neu geordnet, aber die Original-Indizes werden über `point_order` gespeichert.

### 3.2 Kollineare Punkte entfernen

**Zweck**: Entfernt Punkte, die auf einer Linie liegen, bevor das erste Dreieck erstellt wird

**Algorithmus:**
```
index = 0
while index + 2 < len(points):
    area = compute_triangle_area(index, index+1, index+2)
    if abs(area) < EPS:
        // Entferne kollinearen Punkt
        delete points[index]
        delete point_order[index]
        point_unorder = argsort(point_order)  // Neu berechnen
    else:
        break  // Erste drei nicht-kollineare Punkte gefunden

if index > len(points) - 3:
    // Alle Punkte sind kollinear
    return  // Keine Triangulation möglich
```

### 3.3 Erstes Dreieck erstellen

**Algorithmus:**
1. Erstelle Dreieck aus ersten drei Punkten:
   ```
   triangle = [index, index+1, index+2]
   ```
2. Stelle sicher, dass das Dreieck gegen den Uhrzeigersinn orientiert ist:
   ```
   ensure_counter_clockwise(triangle)
   ```
3. Füge Dreieck zur Liste hinzu:
   ```
   triangles.append(triangle)
   tri_idx = 0  // Index des ersten Dreiecks
   ```

### 3.4 Boundary-Edges initialisieren

**Algorithmus:**
1. Erstelle alle drei Edges des ersten Dreiecks (mit Orientierung):
   ```
   e01 = (triangle[0], triangle[1])
   e12 = (triangle[1], triangle[2])
   e20 = (triangle[2], triangle[0])
   ```
2. Füge alle drei Edges zu `boundary_edges` hinzu:
   ```
   boundary_edges.add(e01)
   boundary_edges.add(e12)
   boundary_edges.add(e20)
   ```

**Wichtig**: `boundary_edges` speichert Edges mit Orientierung (als Tupel), NICHT als normalisierte Keys.

### 3.5 Edge- und Punkt-Mappings initialisieren

**Algorithmus:**
1. Für jede Edge des ersten Dreiecks:
   ```
   e01_key = make_edge_key(e01[0], e01[1])
   e12_key = make_edge_key(e12[0], e12[1])
   e20_key = make_edge_key(e20[0], e20[1])
   
   edge_to_triangles[e01_key] = [0]
   edge_to_triangles[e12_key] = [0]
   edge_to_triangles[e20_key] = [0]
   ```
2. Für jeden Punkt des Dreiecks:
   ```
   for each point i in triangle:
       point_to_triangles[i] = {0}
   ```

### 3.6 Weitere Punkte hinzufügen

**Algorithmus:**
```
for i from 3 to len(points) - 1:
    add_point(i)
```

Nach dem Hinzufügen aller Punkte:
```
point_unorder = argsort(point_order)  // Mapping aktualisieren
```

---

## 4. Hilfsfunktionen

### 4.1 Berechnung der signierten Dreiecksfläche

**Funktion: `compute_triangle_area(i0, i1, i2)`**

**Zweck**: Berechnet die signierte Fläche eines Dreiecks (2D-Kreuzprodukt)

**Formel:**
```
d1 = points[i1] - points[i0]
d2 = points[i2] - points[i0]
area = d1.x * d2.y - d1.y * d2.x
```

**Interpretation:**
- `area > 0`: Dreieck ist gegen den Uhrzeigersinn orientiert
- `area < 0`: Dreieck ist im Uhrzeigersinn orientiert
- `abs(area) < EPS`: Punkte sind kollinear

### 4.2 Sichtbarkeit von Edge prüfen

**Funktion: `is_visible_from_edge(point_idx, edge)`**

**Zweck**: Prüft, ob ein Punkt von einer Edge aus "sichtbar" ist (liegt rechts der Edge)

**Algorithmus:**
```
area = compute_triangle_area(point_idx, edge[0], edge[1])
return area < EPS
```

**Geometrische Bedeutung**: Der Punkt liegt rechts der Edge, wenn man entlang der Edge schaut.

### 4.3 Gegen-Uhrzeigersinn-Orientierung sicherstellen

**Funktion: `ensure_counter_clockwise(triangle_indices)`**

**Zweck**: Reordnet die Vertices eines Dreiecks, sodass es gegen den Uhrzeigersinn orientiert ist

**Algorithmus:**
```
area = compute_triangle_area(triangle[0], triangle[1], triangle[2])
if area < -EPS:
    // Dreieck ist im Uhrzeigersinn, vertausche letzte zwei Vertices
    swap(triangle[1], triangle[2])
```

**Wichtig**: Die Funktion modifiziert `triangle_indices` in-place.

### 4.4 Dreieck-Edges extrahieren

**Funktion: `triangle_to_edges(triangle, create_key)`**

**Zweck**: Extrahiert alle drei Edges eines Dreiecks

**Algorithmus:**
```
triangle_cyclic = triangle + [triangle[0]]  // Zyklische Erweiterung
edges = []
for i from 0 to 2:
    edge = (triangle_cyclic[i], triangle_cyclic[i+1])
    if create_key:
        edges.append(make_edge_key(edge[0], edge[1]))
    else:
        edges.append(edge)  // Mit Orientierung
return edges
```

**Wichtig**: 
- Wenn `create_key=true`: Edges werden normalisiert (für Lookups)
- Wenn `create_key=false`: Edges behalten Orientierung (für geometrische Operationen)

### 4.5 Edge-Intersektion prüfen

**Funktion: `edges_intersect(edge1, edge2)`**

**Zweck**: Prüft, ob zwei Edges sich schneiden (Endpunkte ausgeschlossen)

**Algorithmus:**
1. Prüfe, ob Edges gemeinsame Vertices haben:
   ```
   if edge1[0] in edge2 or edge1[1] in edge2:
       return false
   ```
2. Berechne Schnittpunkt:
   ```
   p11 = points[edge1[0]]
   p12 = points[edge1[1]]
   p21 = points[edge2[0]]
   p22 = points[edge2[1]]
   
   t = p12 - p11
   s = p22 - p21
   r = p21 - p11
   
   // Löse: t * c1 - s * c2 = r
   coeffs = solve([t, -s], r)
   c1, c2 = coeffs
   ```
3. Prüfe, ob Schnittpunkt innerhalb beider Edges liegt:
   ```
   return (0 < c1 < 1) and (0 < c2 < 1)
   ```

**Fehlerbehandlung**: Wenn das Gleichungssystem nicht lösbar ist (parallele Edges), return `false`.

---

## 5. Punkt hinzufügen

**Funktion: `add_point(point_idx)`**

**Zweck**: Fügt einen neuen Punkt zur Triangulation hinzu

### 5.1 Vorbereitung

```
edges_to_remove = {}  // Set von Edges, die entfernt werden sollen
edges_to_add = {}  // Set von Edges, die hinzugefügt werden sollen
```

### 5.2 Neue Dreiecke erstellen

**Algorithmus:**
```
for each edge in boundary_edges:
    if is_visible_from_edge(point_idx, edge):
        // Erstelle neues Dreieck
        new_triangle = [edge[0], edge[1], point_idx]
        ensure_counter_clockwise(new_triangle)
        triangles.append(new_triangle)
        tri_idx = len(triangles) - 1
        
        // Update Edge-Mappings
        e0 = make_edge_key(edge[0], edge[1])
        e1 = make_edge_key(point_idx, edge[0])
        e2 = make_edge_key(edge[1], point_idx)
        
        for each e in [e0, e1, e2]:
            if e not in edge_to_triangles:
                edge_to_triangles[e] = []
            edge_to_triangles[e].append(tri_idx)
        
        // Update Punkt-zu-Dreieck-Mappings
        for each i in new_triangle:
            if i not in point_to_triangles:
                point_to_triangles[i] = {}
            point_to_triangles[i].add(tri_idx)
        
        // Track Boundary-Edge-Updates
        edges_to_remove.add(edge)  // Original-Edge mit Orientierung
        edges_to_add.add((edge[0], point_idx))
        edges_to_add.add((point_idx, edge[1]))
```

### 5.3 Boundary-Edges aktualisieren

**Algorithmus:**
```
// Entferne alte Boundary-Edges
for each edge in edges_to_remove:
    boundary_edges.remove(edge)

// Füge neue Boundary-Edges hinzu (nur wenn sie genau einem Dreieck gehören)
for each edge in edges_to_add:
    edge_key = make_edge_key(edge[0], edge[1])
    if len(edge_to_triangles.get(edge_key, [])) == 1:
        boundary_edges.add(edge)  // Mit Orientierung
```

**Wichtig**: 
- `boundary_edges` speichert Edges mit Orientierung (nicht normalisiert)
- Eine Edge ist eine Boundary-Edge, wenn sie genau einem Dreieck gehört

### 5.4 Delaunay-Kriterium erzwingen

**Wenn `delaunay = true`:**
```
flip_edges()  // Siehe Abschnitt 6
```

---

## 6. Edge-Flipping (Delaunay-Triangulation)

### 6.1 Delaunay-Kriterium prüfen

**Funktion: `flip_edge(edge, enforce_delaunay, check_intersection)`**

**Zweck**: Flippt eine Edge zwischen zwei Dreiecken, wenn das Delaunay-Kriterium verletzt ist

**Voraussetzungen:**
- Die Edge muss genau zwei Dreiecken angehören
- `edge` muss bereits normalisiert sein (als Key)

### 6.2 Edge-Flip-Algorithmus

**Schritt 1: Prüfe, ob Edge geflippt werden kann**
```
triangles = edge_to_triangles.get(edge, [])
if len(triangles) < 2:
    return {}  // Edge ist Boundary-Edge oder existiert nicht
```

**Schritt 2: Finde gegenüberliegende Vertices**
```
tri1_idx, tri2_idx = triangles[0], triangles[1]
tri1 = triangles[tri1_idx]
tri2 = triangles[tri2_idx]

opposite1 = None
opposite2 = None

for i from 0 to 2:
    if tri1[i] not in edge:
        opposite1 = tri1[i]
    if tri2[i] not in edge:
        opposite2 = tri2[i]
```

**Schritt 3: Prüfe Intersektion (wenn `check_intersection = true`)**
```
if check_intersection:
    diagonal = make_edge_key(opposite1, opposite2)
    if not edges_intersect(edge, diagonal):
        return {}  // Edges schneiden sich nicht
```

**Schritt 4: Prüfe Delaunay-Kriterium (wenn `enforce_delaunay = true`)**
```
if enforce_delaunay:
    // Berechne Vektoren
    da1 = points[edge[0]] - points[opposite1]
    db1 = points[edge[1]] - points[opposite1]
    da2 = points[edge[0]] - points[opposite2]
    db2 = points[edge[1]] - points[opposite2]
    
    // Berechne Winkel
    cross1 = compute_triangle_area(opposite1, edge[0], edge[1])
    cross2 = compute_triangle_area(opposite2, edge[1], edge[0])
    dot1 = dot(da1, db1)
    dot2 = dot(da2, db2)
    
    angle1 = abs(atan2(cross1, dot1))
    angle2 = abs(atan2(cross2, dot2))
    
    // Delaunay-Kriterium: Flip wenn Summe der gegenüberliegenden Winkel > π
    if not (angle1 + angle2 > π * (1.0 + EPS)):
        return {}  // Delaunay-Kriterium erfüllt, kein Flip nötig
```

**Schritt 5: Flippe die Dreiecke**
```
new_tri1 = [opposite1, edge[0], opposite2]
new_tri2 = [opposite1, opposite2, edge[1]]

triangles[tri1_idx] = new_tri1
triangles[tri2_idx] = new_tri2
```

**Schritt 6: Update Edge-Mappings**
```
// Entferne alte Edge
delete edge_to_triangles[edge]

// Füge neue Edge hinzu
new_edge = make_edge_key(opposite1, opposite2)
edge_to_triangles[new_edge] = [tri1_idx, tri2_idx]

// Update Punkt-zu-Dreieck-Mappings für neue Edge
point_to_triangles[new_edge[0]].add(tri1_idx)
point_to_triangles[new_edge[0]].add(tri2_idx)
point_to_triangles[new_edge[1]].add(tri1_idx)
point_to_triangles[new_edge[1]].add(tri2_idx)
```

**Schritt 7: Update betroffene Edges**
```
// Edge e1: (opposite1, edge[1])
e1 = make_edge_key(opposite1, edge[1])
if e1 in edge_to_triangles:
    tris = edge_to_triangles[e1]
    for i from 0 to len(tris) - 1:
        if tris[i] == tri1_idx:
            tris[i] = tri2_idx  // Update Triangle-Index
    result_edges.add(e1)

// Edge e2: (opposite2, edge[0])
e2 = make_edge_key(opposite2, edge[0])
if e2 in edge_to_triangles:
    tris = edge_to_triangles[e2]
    for i from 0 to len(tris) - 1:
        if tris[i] == tri2_idx:
            tris[i] = tri1_idx  // Update Triangle-Index
    result_edges.add(e2)
```

**Schritt 8: Update Punkt-zu-Dreieck-Mappings für gemeinsame Vertices**
```
// Für Vertices, die in der alten Edge waren
for each i in new_tri1:
    if i in edge:  // i ist einer der Endpunkte der geflippten Edge
        tris_set = list(point_to_triangles[i])
        for j from 0 to len(tris_set) - 1:
            if tris_set[j] == tri2_idx:
                tris_set[j] = tri1_idx  // Update Index
        point_to_triangles[i] = set(tris_set)

for each i in new_tri2:
    if i in edge:
        tris_set = list(point_to_triangles[i])
        for j from 0 to len(tris_set) - 1:
            if tris_set[j] == tri1_idx:
                tris_set[j] = tri2_idx  // Update Index
        point_to_triangles[i] = set(tris_set)
```

**Schritt 9: Rückgabe von Edges, die möglicherweise geflippt werden müssen**
```
result_edges.add(make_edge_key(opposite1, edge[0]))
result_edges.add(make_edge_key(opposite2, edge[1]))
return result_edges
```

### 6.3 Alle Edges flippen

**Funktion: `flip_edges()`**

**Zweck**: Flippt alle Edges, bis das Delaunay-Kriterium erfüllt ist

**Algorithmus:**
```
edge_set = set(edge_to_triangles.keys())

while edge_set is not empty:
    new_edge_set = {}
    for each edge in edge_set:
        result_edges = flip_edge(edge, enforce_delaunay=true)
        new_edge_set.update(result_edges)
    edge_set = new_edge_set
```

**Wichtig**: Der Algorithmus iteriert, bis keine Edges mehr geflippt werden müssen.

---

## 7. Constraint Enforcement

### 7.1 Boundary-Liste erstellen

**Funktion: `create_boundary_list(border_indices, create_key)`**

**Zweck**: Erstellt eine Liste von Boundary-Edges aus Boundary-Definitionen

**Algorithmus:**
```
if boundaries is None:
    return []

boundary_edges = []
for k from 0 to len(boundaries) - 1:
    boundary = boundaries[k]
    
    // Prüfe, ob Boundary verwendet werden soll
    if border_indices is not None and k not in border_indices:
        continue
    
    // Mappe Boundary-Indizes von original zu sortiert
    boundary_original = point_unorder[boundary]
    
    // Erstelle Edges aus aufeinanderfolgenden Punkten
    for i from 0 to len(boundary_original) - 2:
        j = i + 1
        edge_original = (boundary_original[i], boundary_original[j])
        
        if create_key:
            boundary_edges.append(make_edge_key(edge_original[0], edge_original[1]))
        else:
            boundary_edges.append(edge_original)  // Mit Orientierung

return boundary_edges
```

**Wichtig**: 
- Wenn `border_indices` leer ist (`[]`), werden ALLE Boundaries verwendet (nicht `None`!)
- Die Boundary-Indizes werden von original zu sortiert gemappt über `point_unorder`
- Die letzte Edge der Boundary wird NICHT explizit erstellt (kein geschlossener Loop)

### 7.2 Edge constrainten

**Funktion: `constrain_edge(edge)`**

**Zweck**: Erzwingt, dass eine Edge in der Triangulation vorhanden ist

**Algorithmus:**
```
edge_key = make_edge_key(edge[0], edge[1])

// Prüfe, ob Edge bereits existiert
if edge_key in edge_to_triangles:
    return  // Nichts zu tun

pt0, pt1 = edge[0], edge[1]

// Validiere Edge-Endpunkte
if pt0 not in point_to_triangles or pt1 not in point_to_triangles:
    raise Error("Edge endpoints are not valid point indices")

// Finde erste schneidende Edge
intersecting_edge = None
for each tri_idx in point_to_triangles.get(pt1, {}):
    tri_vertices = list(triangles[tri_idx])
    if pt1 in tri_vertices:
        tri_vertices.remove(pt1)
        if len(tri_vertices) == 2:
            candidate_edge = make_edge_key(tri_vertices[0], tri_vertices[1])
            if edges_intersect(candidate_edge, edge_key):
                intersecting_edge = candidate_edge
                break

if intersecting_edge is None:
    return  // Edge bereits constraint oder keine Intersektion gefunden

// Flippe Edges, bis Constraint erfüllt ist
edges_to_check = flip_edge(intersecting_edge, 
                          enforce_delaunay=false, 
                          check_intersection=true)

while true:
    found_intersection = false
    for each e in edges_to_check:
        if edges_intersect(e, edge_key):
            intersecting_edge = e
            found_intersection = true
            break
    
    if not found_intersection:
        break
    
    edges_to_check = flip_edge(intersecting_edge,
                              enforce_delaunay=false,
                              check_intersection=true)
    
    if edges_to_check is empty:
        if edge_key in edge_to_triangles:
            break
        else:
            // Rekursiv constraint Sub-Edges
            constrain_edge(make_edge_key(intersecting_edge[0], pt0))
            constrain_edge(make_edge_key(pt0, pt1))
```

**Wichtig**: 
- Wenn keine schneidende Edge gefunden wird, wird rekursiv mit Sub-Edges gearbeitet
- `enforce_delaunay=false`, da wir nur die Constraint-Erfüllung wollen

### 7.3 Alle Boundaries constrainten

**Funktion: `constrain_boundaries()`**

**Algorithmus:**
```
boundary_edges = create_boundary_list(create_key=true)
for each edge in boundary_edges:
    constrain_edge(edge)
```

---

## 8. Mappings aktualisieren

**Funktion: `update_mappings()`**

**Zweck**: Baut Edge-zu-Dreieck- und Punkt-zu-Dreieck-Mappings neu auf

**Algorithmus:**
```
// Lösche alte Mappings
edge_to_triangles = {}
point_to_triangles = {}

// Baue Mappings neu auf
for tri_idx from 0 to len(triangles) - 1:
    triangle = triangles[tri_idx]
    
    // Edge-zu-Dreieck-Mapping
    for each edge in triangle_to_edges(triangle, create_key=true):
        if edge not in edge_to_triangles:
            edge_to_triangles[edge] = []
        edge_to_triangles[edge].append(tri_idx)
    
    // Punkt-zu-Dreieck-Mapping
    for each point_idx in triangle:
        if point_idx not in point_to_triangles:
            point_to_triangles[point_idx] = {}
        point_to_triangles[point_idx].add(tri_idx)

// Update Boundary-Edges
// Boundary-Edges sind Edges, die genau einem Dreieck angehören
new_boundary_edges = {}
for each edge, triangles in edge_to_triangles.items():
    if len(triangles) == 1:
        new_boundary_edges.add(edge)

boundary_edges = new_boundary_edges
```

**Wichtig**: 
- `boundary_edges` wird komplett neu berechnet
- Eine Edge ist eine Boundary-Edge, wenn sie genau einem Dreieck angehört
- `boundary_edges` speichert normalisierte Edge-Keys (nicht Tupel mit Orientierung)

---

## 9. Leere Dreiecke entfernen

**Funktion: `remove_empty_triangles()`**

**Zweck**: Entfernt Dreiecke mit null oder nahezu null Fläche

**Algorithmus:**
```
triangles_to_remove = []
for i from 0 to len(triangles) - 1:
    triangle = triangles[i]
    ensure_counter_clockwise(triangle)
    area = compute_triangle_area(triangle[0], triangle[1], triangle[2])
    if abs(area) < EPS:
        triangles_to_remove.append(i)

// Entferne in umgekehrter Reihenfolge (damit Indizes stabil bleiben)
if triangles_to_remove is not empty:
    sort triangles_to_remove in descending order
    for each i in triangles_to_remove:
        delete triangles[i]
```

**Wichtig**: Dreiecke werden in umgekehrter Reihenfolge entfernt, damit die Indizes der verbleibenden Dreiecke stabil bleiben.

---

## 10. Hole Removal

**Funktion: `remove_holes()`**

**Zweck**: Entfernt Dreiecke innerhalb von Löchern, die durch Boundaries definiert sind

### 10.1 Vorbereitung

**Algorithmus:**
```
if boundaries is None:
    return

// Erstelle Boundary-Listen
// WICHTIG: Wenn border leer ist ([]), verwende None für alle Boundaries
border_indices = (border is empty) ? None : border

boundary_keys = create_boundary_list(border_indices, create_key=true)
boundary_tuples = create_boundary_list(border_indices, create_key=false)
```

**Wichtig**: 
- Wenn `border` leer ist (`[]`), wird `None` übergeben, was bedeutet, dass ALLE Boundaries verwendet werden
- `boundary_keys`: Normalisierte Edge-Keys für Lookups
- `boundary_tuples`: Edges mit Orientierung für Vergleich

### 10.2 Start-Dreiecke identifizieren

**Algorithmus:**
```
edges_to_remove = {}

for each (b_key, b_tuple) in zip(boundary_keys, boundary_tuples):
    triangles = edge_to_triangles.get(b_key, [])
    for each tri_idx in triangles:
        tri_edges = triangle_to_edges(triangles[tri_idx], create_key=false)
        
        // Prüfe, ob Dreieck diese Boundary-Edge enthält (mit Orientierung)
        if b_tuple in tri_edges:
            // Markiere ALLE Edges dieses Dreiecks zur Entfernung
            for each edge in triangle_to_edges(triangles[tri_idx], create_key=true):
                edges_to_remove.add(edge)
```

**Wichtig**: 
- Es wird geprüft, ob `b_tuple` (mit Orientierung) in `tri_edges` ist
- Wenn ein Dreieck eine Boundary-Edge enthält, werden ALLE drei Edges des Dreiecks markiert

### 10.3 Boundary-Edges schützen

**Algorithmus:**
```
// Entferne Boundary-Edges selbst aus der Entfernung-Liste
for each b_key in boundary_keys:
    edges_to_remove.remove(b_key)
```

### 10.4 Initiale Dreiecke sammeln

**Algorithmus:**
```
triangles_to_remove = {}
for each edge in edges_to_remove:
    triangles_to_remove.update(edge_to_triangles.get(edge, []))
```

### 10.5 Iterative Expansion (Flood-Fill)

**Algorithmus:**
```
prev_count = len(triangles_to_remove)

while true:
    // Erweitere edges_to_remove um alle Edges der markierten Dreiecke
    for each tri_idx in triangles_to_remove:
        for each edge in triangle_to_edges(triangles[tri_idx], create_key=true):
            edges_to_remove.add(edge)
    
    // Schütze Boundary-Edges erneut
    for each b_key in boundary_keys:
        edges_to_remove.remove(b_key)
    
    // Finde alle neuen Dreiecke, die diese Edges enthalten
    for each edge in edges_to_remove:
        triangles_to_remove.update(edge_to_triangles.get(edge, []))
    
    // Prüfe, ob noch neue Dreiecke gefunden wurden
    if len(triangles_to_remove) == prev_count:
        break  // Keine neuen Dreiecke mehr
    
    prev_count = len(triangles_to_remove)
```

**Wichtig**: 
- Der Algorithmus expandiert iterativ von den Start-Dreiecken
- Boundary-Edges werden in jeder Iteration geschützt
- Der Algorithmus stoppt, wenn keine neuen Dreiecke mehr hinzugefügt werden

### 10.6 Dreiecke entfernen

**Algorithmus:**
```
if triangles_to_remove is not empty:
    sort triangles_to_remove in descending order
    for each i in triangles_to_remove:
        delete triangles[i]
```

**Wichtig**: Dreiecke werden in umgekehrter Reihenfolge entfernt, damit Indizes stabil bleiben.

---

## 11. Hauptalgorithmus

**Funktion: `PolyTri(points, boundaries, delaunay, holes, border)`**

**Vollständiger Ablauf:**

1. **Validierung**: Prüfe Eingabeparameter
2. **Initialisierung**: Erstelle leere Datenstrukturen
3. **Triangulation-Initialisierung**: 
   - Sortiere Punkte
   - Entferne kollineare Punkte
   - Erstelle erstes Dreieck
   - Initialisiere Mappings
   - Füge alle weiteren Punkte hinzu
4. **Constraints anwenden** (wenn `boundaries` gegeben):
   - `constrain_boundaries()`
   - Wenn `holes = true`:
     - `remove_empty_triangles()`
     - `update_mappings()`
     - `remove_holes()`

---

## 12. Ausgabe

### 12.1 Dreiecke abrufen

**Funktion: `get_triangles()`**

**Zweck**: Gibt Dreiecke als Arrays von originalen Punkt-Indizes zurück

**Algorithmus:**
```
result = []
for each triangle in triangles:
    // Mappe von sortierten zu originalen Indizes
    original_triangle = [point_order[i] for i in triangle]
    result.append(original_triangle)
return result
```

**Wichtig**: Die Dreiecke werden von internen (sortierten) Indizes zu originalen Indizes gemappt.

### 12.2 Boundary-Edges abrufen

**Eigenschaft: `boundary_edges`**

**Wichtig**: 
- `boundary_edges` enthält normalisierte Edge-Keys (nicht Tupel mit Orientierung)
- Diese müssen für die Ausgabe möglicherweise zu originalen Indizes gemappt werden

---

## 13. Wichtige Implementierungsdetails

### 13.1 Edge-Normalisierung

- **Immer verwenden für**: Lookups in `edge_to_triangles`
- **Nicht verwenden für**: Geometrische Operationen, Boundary-Edge-Speicherung (in `_boundary_edges`)

### 13.2 Index-Mapping

- **Intern**: Alle Operationen verwenden sortierte Indizes
- **Extern**: Alle Ausgaben verwenden originale Indizes über `point_order`

### 13.3 Boundary-Edges

- **In `_boundary_edges`**: Gespeichert als Tupel mit Orientierung (für `add_point`)
- **In `boundary_edges` (Property)**: Gespeichert als normalisierte Keys (nach `update_mappings`)

### 13.4 Leere Listen vs. None

- **`border = []`**: Leere Liste bedeutet "verwende alle Boundaries" (wird zu `None` konvertiert)
- **`border = None`**: Wird zu `[]` konvertiert, dann zu `None` für `create_boundary_list`

### 13.5 Geschlossene Boundaries

- Boundaries sind geschlossene Loops, aber die letzte Edge wird nicht explizit erstellt
- `zip(boundary_original[:-1], boundary_original[1:])` erstellt Edges zwischen aufeinanderfolgenden Punkten

### 13.6 Delaunay nach Punkt-Hinzufügung

- Nach jedem `add_point()` wird `flip_edges()` aufgerufen (wenn `delaunay = true`)
- Dies stellt sicher, dass die Triangulation nach jedem Punkt Delaunay-konform ist

### 13.7 Mappings nach großen Änderungen

- Nach `remove_holes()` oder `remove_empty_triangles()` sollte `update_mappings()` aufgerufen werden
- Dies stellt sicher, dass alle Mappings konsistent sind

---

## 14. Fehlerbehandlung

### 14.1 Kollineare Punkte

- Werden automatisch entfernt während der Initialisierung
- Wenn alle Punkte kollinear sind, wird keine Triangulation erstellt

### 14.2 Ungültige Edge-Endpunkte

- Werden während `constrain_edge()` validiert
- Fehler wird geworfen, wenn Endpunkte nicht in `point_to_triangles` existieren

### 14.3 Edge-Intersektion-Berechnung

- Wenn das Gleichungssystem nicht lösbar ist (parallele Edges), wird `false` zurückgegeben
- Dies wird in `edges_intersect()` behandelt

---

## 15. Performance-Hinweise

### 15.1 Datenstrukturen

- **Sets**: Für `boundary_edges`, `triangles_to_remove` (schnelle Lookups)
- **Lists**: Für `triangles`, `edge_to_triangles[edge]` (Reihenfolge wichtig)
- **Dictionaries**: Für `edge_to_triangles`, `point_to_triangles` (schnelle Lookups)

### 15.2 Edge-Normalisierung

- Wird bei jedem Edge-Lookup durchgeführt
- Sollte gecacht werden, wenn Performance kritisch ist

### 15.3 Mappings aktualisieren

- Wird nach großen Änderungen benötigt
- Kann teuer sein für große Triangulationen

---

## Ende der Dokumentation

Diese Beschreibung enthält alle Details des Algorithmus, einschließlich impliziter Operationen, die Python automatisch durchführt. Sie sollte ausreichen, um den Algorithmus in jeder Programmiersprache zu implementieren.

