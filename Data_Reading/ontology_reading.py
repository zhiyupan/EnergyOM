import os
import pandas as pd
from owlready2 import get_ontology, World, ThingClass
from Data_Processing.utilizing import get_file_path
from collections import deque
from rdflib import Graph, RDF, RDFS, URIRef, BNode
import re

save_folder = '../Data/onto_data'


class OntologyReading(object):
    def __init__(self, path):
        """
        Initializes the OntologyReading instance with the path to the ontology file
        :param path: the file path to the ontology file to loaded.
        """
        self.path = path
        self._onto = None

    @property
    def name(self):
        # Return the name of the ontology file without the file extension.
        return os.path.basename(self.path).split('.')[0]

    def onto(self):
        if self._onto is not None:
            return self._onto
        try:
            self._onto_cache = get_ontology(self.path).load()
            print(f"Ontology: {self.name} loaded successfully!")
            return self._onto_cache
        except Exception as e:
            print(f"Error loading ontology in strict mode: {e}")
            print("Attempting to load in lenient mode...")
            try:
                world = World()
                world.graph.strict = False
                self._onto_cache = world.get_ontology(self.path).load()
                print("Ontology loaded successfully in lenient mode!")
                return self._onto_cache
            except Exception as lenient_e:
                print(f"Failed to load ontology even in lenient mode: {lenient_e}")
                print("Attempting to skip problematic entities...")
                world = World()
                self._onto_cache = world.get_ontology(self.path)
                problem_entities = set()

                for cls in self._onto_cache.classes():
                    try:
                        cls.iri
                    except Exception as cls_e:
                        print(f"Problematic entity detected: {cls}, due to {cls_e}")
                        problem_entities.add(cls)
                print(f"Skipping {len(problem_entities)} problematic entities.")
                for entity in problem_entities:
                    entity.destroy()

                self._onto = world.get_ontology(self.path).load()
                print("Ontology reloaded after skipping problematic entities.")
                return self._onto_cache

    def _get_label_or_name(self, cls):
        if hasattr(cls, 'label') and cls.label:
            return str(cls.label[0])
        return cls.name

    def get_classes(
            self,
            include_imports: bool = True,
            import_dirs: list = None,
            import_map: dict = None,
            fetch_online: bool = True,
            prefer_label_langs=("en", "de", "zh"),
            return_iri: bool = False,
            debug: bool = False
    ):
        """
        文件级递归解析（主文件 + imports），仅收“命名类”：
          - 排除：数据类型（xsd:*, rdfs:Literal）、属性 IRI、oneOf 的个体成员、各种 Restriction 的数据类型填充。
          - 仅在 domain/range 中：只收“类一侧”的对象；range 仅当属性是 ObjectProperty 才收类。
          - 其它同前：优先 rdfs:label（按语言），否则 IRI 本地名。
        """
        import os, io
        from collections import deque, defaultdict

        # 可用 rdflib 时支持 RDF/XML、TTL、NT、N3
        try:
            from rdflib import Graph, URIRef, BNode, RDF, RDFS, Literal, Namespace
            HAS_RDFLIB = True
        except Exception:
            HAS_RDFLIB = False

        # 可用 requests 时支持在线抓取
        try:
            import requests
            HAS_REQUESTS = True
        except Exception:
            HAS_REQUESTS = False

        from xml.etree import ElementTree as ET

        OWL_NS = "http://www.w3.org/2002/07/owl#"
        RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
        XSD_NS = "http://www.w3.org/2001/XMLSchema#"

        import_dirs = import_dirs or []
        import_map = import_map or {}

        # --------------- 小工具 --------------- #
        def _local_name(iri: str) -> str:
            s = str(iri)
            if "#" in s: return s.rsplit("#", 1)[-1] or s
            if "/" in s: return s.rstrip("/").rsplit("/", 1)[-1] or s
            return s

        def _is_owlxml_bytes(head: bytes) -> bool:
            try:
                h = head.decode("utf-8", errors="ignore")
            except Exception:
                return False
            return ("<Ontology" in h and OWL_NS in h and "<rdf:RDF" not in h)

        def _is_owlxml_file(path: str) -> bool:
            try:
                with open(path, "rb") as f:
                    return _is_owlxml_bytes(f.read(4096))
            except Exception:
                return False

        def _guess_rdflib_format(path_or_url: str):
            s = path_or_url.lower()
            if s.endswith(".ttl"): return "turtle"
            if s.endswith(".rdf"): return "xml"
            if s.endswith(".owl"): return None
            if s.endswith(".nt"):  return "nt"
            if s.endswith(".n3"):  return "n3"
            if s.endswith(".xml"): return "xml"
            return None

        # —— 判定：是不是“数据类型 IRI”？—— #
        def _is_datatype_uri(u) -> bool:
            su = str(u)
            if su.startswith(XSD_NS): return True  # xsd:*
            if su == (RDFS_NS + "Literal"): return True  # rdfs:Literal
            # 其他已知数据类型命名空间（按需可再补充）
            return False

        # —— 仅当对象确实像“类”时才加入 —— #
        def _add_if_classlike(target_set: set, u):
            if isinstance(u, URIRef) and not _is_datatype_uri(u):
                target_set.add(str(u))

        # --------------- RDF/XML / Turtle （rdflib）解析 --------------- #
        def _read_with_rdflib(source_path_or_url: str, content_bytes: bytes = None):
            """
            返回 (classes:set[str], labels:dict[str,list[(txt,lang)]], import_iris:set[str])
            ——严格只收“类”，过滤掉数据类型与属性 IRI——
            """
            classes, labels, imports = set(), defaultdict(list), set()
            if not HAS_RDFLIB:
                return classes, labels, imports

            g = Graph()
            fmt = _guess_rdflib_format(source_path_or_url)
            if content_bytes is None:
                g.parse(source_path_or_url, format=fmt)
            else:
                g.parse(io.BytesIO(content_bytes), format=fmt)

            OWL = Namespace(OWL_NS)

            # 先收集属性类型（用于判断 range 是否类）
            obj_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
            dt_props = set(g.subjects(RDF.type, OWL.DatatypeProperty))

            # 1) 明确类声明
            for s in g.subjects(RDF.type, OWL.Class):
                _add_if_classlike(classes, s)
            for s in g.subjects(RDF.type, RDFS.Class):
                _add_if_classlike(classes, s)

            # 2) 子类/等价/互斥：两侧通常是类
            for p in (RDFS.subClassOf, OWL.equivalentClass, OWL.disjointWith):
                for s, o in g.subject_objects(p):
                    _add_if_classlike(classes, s)
                    _add_if_classlike(classes, o)

            # 3) domain/range：
            #    - 只看“对象 o”这一侧（类），绝不把“属性 s”当类加入
            #    - range：仅当属性是 ObjectProperty；若属性未声明类型，则启发式排除 xsd/rdfs:Literal
            for s, o in g.subject_objects(RDFS.domain):
                _add_if_classlike(classes, o)  # domain 对 DatatypeProperty 也是类

            for s, o in g.subject_objects(RDFS.range):
                if s in dt_props:
                    # DatatypeProperty 的 range 是数据类型：跳过
                    continue
                # ObjectProperty 或未知类型——仅当不是数据类型时加入
                if isinstance(o, URIRef) and not _is_datatype_uri(o):
                    classes.add(str(o))

            # 4) 构造式：unionOf / intersectionOf（RDF 列表的成员有时是类表达式）
            RDF_FIRST, RDF_REST, RDF_NIL = RDF.first, RDF.rest, RDF.nil

            def rdf_list_members(node):
                out, cur, seen = [], node, set()
                while cur and cur != RDF_NIL and cur not in seen:
                    seen.add(cur)
                    first = next(g.objects(cur, RDF_FIRST), None)
                    if first is not None:
                        out.append(first)
                    cur = next(g.objects(cur, RDF_REST), None)
                    if cur is None:
                        break
                return out

            for ctor in (OWL.unionOf, OWL.intersectionOf):
                for n in set(g.subjects(ctor, None)):
                    lst = next(g.objects(n, ctor), None)
                    if lst is not None:
                        for m in rdf_list_members(lst):
                            _add_if_classlike(classes, m)

            # 注意：oneOf 是“个体枚举”，成员不是类——不要加！
            # for ctor in (OWL.oneOf,):  # 故意不处理成员

            # 5) 限制：onClass / someValuesFrom / allValuesFrom / hasValue / complementOf / *Cardinality
            #    只要“填充”看起来像类（不是 xsd/rdfs:Literal），才加入
            for ctor in (OWL.onClass, OWL.someValuesFrom, OWL.allValuesFrom, OWL.hasValue,
                         OWL.complementOf, OWL.ObjectMinCardinality, OWL.ObjectMaxCardinality,
                         OWL.ObjectExactCardinality):
                for n in set(g.subjects(ctor, None)):
                    for m in g.objects(n, ctor):
                        _add_if_classlike(classes, m)

            # 6) rdfs:label（只给已收的类加标签）
            for iri in list(classes):
                for lab in g.objects(URIRef(iri), RDFS.label):
                    if isinstance(lab, Literal):
                        labels[iri].append(
                            (str(lab), (getattr(lab, "language", None) or getattr(lab, "lang", None) or "").lower()))

            # 7) imports
            for o in g.objects(None, OWL.imports):
                if isinstance(o, URIRef):
                    imports.add(str(o))

            # 8) 去掉 owl:Thing / owl:Nothing（如不想去可注释）
            classes.discard(OWL_NS + "Thing")
            classes.discard(OWL_NS + "Nothing")

            return classes, labels, imports

        # --------------- OWL/XML（ElementTree）解析 --------------- #
        # 这个分支本来就只抓 <owl:Class ...>，不会把 xsd 数据类型当类，因此无需改动太多
        def _read_owlxml(path: str, content_bytes: bytes = None):
            ns = {"owl": OWL_NS, "rdfs": RDFS_NS, "rdf": RDF_NS, "xml": "http://www.w3.org/XML/1998/namespace"}
            classes, labels, imports = set(), defaultdict(list), set()
            root = ET.fromstring(content_bytes) if content_bytes is not None else ET.parse(path).getroot()

            def add_class_el(el):
                iri = el.attrib.get("IRI") or el.attrib.get("abbreviatedIRI")
                if iri and (iri.startswith("http://") or iri.startswith("https://")):
                    classes.add(iri)

            # Class 声明与各处出现的 Class
            for e in root.findall(
                    ".//{http://www.w3.org/2002/07/owl#}Declaration/{http://www.w3.org/2002/07/owl#}Class"):
                add_class_el(e)
            for e in root.findall(".//{http://www.w3.org/2002/07/owl#}Class"):
                add_class_el(e)

            # 公理与构造式中的 Class
            complex_paths = [
                ".//owl:SubClassOf//owl:Class",
                ".//owl:EquivalentClasses//owl:Class",
                ".//owl:DisjointClasses//owl:Class",
                ".//owl:ObjectPropertyDomain//owl:Class",
                ".//owl:ObjectPropertyRange//owl:Class",
                ".//owl:ObjectUnionOf//owl:Class",
                ".//owl:ObjectIntersectionOf//owl:Class",
                ".//owl:ObjectComplementOf//owl:Class",
                ".//owl:ObjectSomeValuesFrom//owl:Class",
                ".//owl:ObjectAllValuesFrom//owl:Class",
                ".//owl:ObjectHasValue//owl:Class",
                ".//owl:ObjectMinCardinality//owl:Class",
                ".//owl:ObjectMaxCardinality//owl:Class",
                ".//owl:ObjectExactCardinality//owl:Class",
            ]
            for p in complex_paths:
                for c in root.findall(p, ns):
                    add_class_el(c)

            # AnnotationAssertion(label)
            for ann in root.findall(".//{http://www.w3.org/2002/07/owl#}AnnotationAssertion"):
                prop = ann.find("./{http://www.w3.org/2002/07/owl#}AnnotationProperty")
                if prop is None:
                    continue
                piri = prop.attrib.get("IRI") or prop.attrib.get("abbreviatedIRI") or ""
                if not (piri.endswith("#label") or piri.endswith("/label")):
                    continue
                subj = ann.find("./{http://www.w3.org/2002/07/owl#}IRI")
                lit = ann.find("./{http://www.w3.org/2002/07/owl#}Literal")
                if subj is not None and subj.text and lit is not None:
                    iri = subj.text.strip()
                    if iri.startswith("http://") or iri.startswith("https://"):
                        lang = (lit.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "")).lower()
                        labels[iri].append(((lit.text or "").strip(), lang))
                        classes.add(iri)

            # imports
            for imp in root.findall(".//{http://www.w3.org/2002/07/owl#}Import"):
                iri = imp.attrib.get("IRI")
                if iri:
                    imports.add(iri)

            # 去掉 Thing/Nothing（如不想去可注释）
            classes.discard(OWL_NS + "Thing")
            classes.discard(OWL_NS + "Nothing")

            return classes, labels, imports

        # --------------- 根据 IRI 找本地/在线 --------------- #
        def _resolve_import_iri(iri: str):
            # 本地映射
            if iri in import_map and os.path.isfile(import_map[iri]):
                return ("file", import_map[iri], None)
            # 本地目录猜测
            last = iri.rstrip("/").rsplit("/", 1)[-1]
            for d in import_dirs:
                for ext in (".owl", ".rdf", ".ttl", ".xml", ".nt", ".n3"):
                    p = os.path.join(d, last + ext)
                    if os.path.isfile(p):
                        return ("file", p, None)
                for ext in (".owl", ".rdf", ".ttl", ".xml", ".nt", ".n3"):
                    p = os.path.join(d, "ontology" + ext)
                    if os.path.isfile(p):
                        return ("file", p, None)
            # 在线
            if fetch_online and HAS_REQUESTS:
                try:
                    r = requests.get(iri, timeout=20)
                    if r.status_code == 200 and r.content:
                        return ("bytes", iri, r.content)
                except Exception:
                    pass
            return (None, None, None)

        # --------------- 递归解析：主文件 + imports --------------- #
        start_path = os.path.abspath(self.path)
        queue = deque([("file", start_path, None)])
        seen_sources = set()
        all_classes, label_map, source_of = set(), defaultdict(list), {}

        while queue:
            src_kind, path_or_url, payload = queue.popleft()
            key = (src_kind, path_or_url)
            if key in seen_sources:
                continue
            seen_sources.add(key)
            if debug:
                print(f"[load] {src_kind}: {path_or_url}")

            try:
                if src_kind == "file":
                    is_owlxml = _is_owlxml_file(path_or_url)
                    if not is_owlxml and HAS_RDFLIB:
                        try:
                            cls_set, labs, imps = _read_with_rdflib(path_or_url)
                        except Exception:
                            is_owlxml = True
                    if is_owlxml:
                        cls_set, labs, imps = _read_owlxml(path_or_url)
                elif src_kind == "bytes":
                    head = payload[:4096] if payload else b""
                    if not _is_owlxml_bytes(head) and HAS_RDFLIB:
                        try:
                            cls_set, labs, imps = _read_with_rdflib(path_or_url, payload)
                        except Exception:
                            cls_set, labs, imps = _read_owlxml(path_or_url, payload)
                    else:
                        cls_set, labs, imps = _read_owlxml(path_or_url, payload)
                else:
                    continue
            except Exception as e:
                if debug:
                    print(f"[load] FAIL: {path_or_url} -> {e}")
                continue

            # 合并
            for iri in cls_set:
                all_classes.add(iri)
                source_of.setdefault(iri, path_or_url)
            for iri, labs in labs.items():
                label_map[iri].extend(labs)
                source_of.setdefault(iri, path_or_url)

            if include_imports:
                for imp_iri in imps:
                    kind, loc, bytes_ = _resolve_import_iri(imp_iri)
                    if kind is None:
                        if debug:
                            print(f"[imports] missing: {imp_iri}")
                        continue
                    queue.append((kind, loc, bytes_))

        # --------------- 映射名字（优先 label） --------------- #
        def _best_label(iri: str) -> str:
            cands = label_map.get(iri, [])
            if cands:
                for lang in prefer_label_langs:
                    for txt, lg in cands:
                        if (lg or "").lower() == lang:
                            return txt
                return cands[0][0]
            return _local_name(iri)

        iris_sorted = sorted(all_classes)
        names = [_best_label(iri) for iri in iris_sorted]

        if debug:
            print(
                f"[get_classes:STRICT-CLASS] files={len({v for v in source_of.values() if v})}  classes={len(iris_sorted)}")
            # 展示前 12 条
            for i, iri in enumerate(iris_sorted[:12]):
                print(f"  {i + 1:02d}. {names[i]}  <{iri}>  @ {source_of.get(iri, '')}")

        return list(zip(names, iris_sorted, [source_of.get(iri) for iri in iris_sorted])) if return_iri else names

    def get_data_properties(self):
        """
        Retrieves a list of data property names from ontology.
        :return: list of str: A list of all data property names in the ontology.
        """
        data_property_list = []
        # Loop through each data property in the ontology
        for i in list(self.onto().data_properties()):
            data_property_name = str(i).split('.')[-1]
            data_property_list.append(data_property_name)
        data_property_list = list(set(data_property_list))
        return data_property_list

    def get_object_properties(self):
        """
        Retrieves a list of object property names from ontology.
        :return: list of str: A list of all object property names in the ontology.
        """
        object_property_list = []
        # Loop through each object property in the ontology
        for i in list(self.onto().object_properties()):
            object_property_name = str(i).split('.')[-1]
            object_property_list.append(object_property_name)
        object_property_list = list(set(object_property_list))
        return object_property_list

    def get_onto_hierarchy_triples(
            self,
            onto_name,
            results_folder=save_folder + '/hierarchy',
            include_imports: bool = False,  # 默认不包含 imports
            only_base_iri: bool = True  # 仅保留当前本体命名空间（需 self.base_iri 或 self.onto_iri）
    ):
        import os
        import pandas as pd
        from collections import deque
        from owlready2 import ThingClass

        def _local_name(entity) -> str:
            iri = getattr(entity, "iri", None)
            if iri:
                s = str(iri)
                if "#" in s: return s.rsplit("#", 1)[-1]
                if "/" in s: return s.rstrip("/").rsplit("/", 1)[-1]
            return getattr(entity, "name", str(entity))

        onto = self.onto()
        base_iri = getattr(self, "base_iri", None) or getattr(self, "onto_iri", None)

        def _in_scope(c) -> bool:
            # 只要“当前本体的命名类”
            if not isinstance(c, ThingClass):  # 排除 Restriction / 构造式
                return False
            if getattr(c, "name", None) == "Thing":
                return False
            try:
                # 优先用 namespace 归属判定
                if c.namespace and getattr(c.namespace, "ontology", None) is onto:
                    return True
            except Exception:
                pass
            # 兜底：按 IRI 前缀筛（需要提供 base_iri）
            if only_base_iri and base_iri:
                iri = getattr(c, "iri", None)
                return bool(iri) and str(iri).startswith(str(base_iri))
            return False  # 没有任何依据则不纳入

        def _key(c):
            return getattr(c, "storid", None) or id(c)

        visited = set()
        edges = set()
        q = deque()

        # 种子：仅当前本体里的类
        for cls in onto.classes():
            if _in_scope(cls):
                k = _key(cls)
                if k not in visited:
                    visited.add(k);
                    q.append(cls)

        # 可选：包含 imports（仍然会经过 _in_scope 二次过滤）
        if include_imports:
            for imp in onto.imported_ontologies:
                for cls in imp.classes():
                    if _in_scope(cls):
                        k = _key(cls)
                        if k not in visited:
                            visited.add(k);
                            q.append(cls)

        while q:
            cls = q.popleft()
            c_name = _local_name(cls)

            # 父类（跨本体返回时依旧用 _in_scope 过滤）
            for parent in cls.is_a:
                if _in_scope(parent):
                    edges.add((c_name, 'is a subclass of', _local_name(parent)))
                    pk = _key(parent)
                    if pk not in visited:
                        visited.add(pk);
                        q.append(parent)

            # 子类（也过滤）
            for sub in cls.subclasses():
                if _in_scope(sub):
                    edges.add((_local_name(sub), 'is a subclass of', c_name))
                    sk = _key(sub)
                    if sk not in visited:
                        visited.add(sk);
                        q.append(sub)

        # 输出
        df = pd.DataFrame(sorted(edges), columns=['Children Class', 'Relationship', 'Parents Class'])
        os.makedirs(results_folder, exist_ok=True)
        suffix = '_ontology_hierarchy_triples.xlsx'
        output_path = get_file_path(results_folder, onto_name, suffix)
        df.to_excel(output_path, index=False)
        return edges

    def get_class_comment_triples(
            self,
            onto_name,
            results_folder=save_folder + '/comment',
            only_base_iri=True,  # 只保留本体自己的命名空间（若你维护了 self.base_iri / self.onto_iri）
            promote_anonymous=True,  # ★ 遇到匿名节点时，将其注释“归属”到引用它的命名类
            promote_max_targets=3  # 一个匿名限制被多个命名类引用时，最多归属几个（防爆表）
    ):
        """
        只解析当前文件 self.path（不依赖 onto()），读取 rdfs:comment，
        输出为 (ClassLocalName, 'comment', Comment)。遇到匿名节点（BNode）时，
        将注释归属到“引用该匿名节点作为 rdfs:subClassOf 目标”的命名类上（promote）。
        """

        def _local_name(x: str) -> str:
            s = str(x)
            if '#' in s:
                return s.rsplit('#', 1)[-1] or s
            if '/' in s:
                return s.rstrip('/').rsplit('/', 1)[-1] or s
            return s

        os.makedirs(results_folder, exist_ok=True)
        suffix = '_ontology_class_comment_triples.xlsx'
        output_path = get_file_path(results_folder, onto_name, suffix)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        g = Graph()
        g.parse(self.path)  # 只读该文件，避免 imports 串味

        # 本体前缀（可选）
        base_iri = getattr(self, "base_iri", None) or getattr(self, "onto_iri", None)

        def _keep(uri: URIRef) -> bool:
            if not only_base_iri or not base_iri:
                return True
            return str(uri).startswith(str(base_iri))

        triples_out = set()

        # 1) 直接处理“命名类”的注释
        #    类的判定：owl:Class / rdfs:Class；另外也有很多类没有显式 type，但有 rdfs:comment
        OWL_CLASS = URIRef("http://www.w3.org/2002/07/owl#Class")
        class_nodes_named = set(g.subjects(RDF.type, OWL_CLASS)) | set(g.subjects(RDF.type, RDFS.Class))

        # 除了已知类，再找所有带注释的 subject
        subjects_with_comment = set(g.subjects(RDFS.comment, None))

        # 并集遍历：所有“可能的类主体”
        candidate_subjects = class_nodes_named | subjects_with_comment

        for s in candidate_subjects:
            # 命名类：直接导出
            if isinstance(s, URIRef):
                if not _keep(s):
                    continue
                comments = list(g.objects(s, RDFS.comment))
                class_name = _local_name(s)
                if comments:
                    for cm in comments:
                        triples_out.add((class_name, 'comment', str(cm)))
                else:
                    # 没注释也留占位，与你原逻辑一致
                    triples_out.add((class_name, 'comment', 'None'))

        # 2) 处理“匿名节点”的注释：把匿名的注释归属给引用它的命名类
        #    常见形态：某命名类 A 有 (A rdfs:subClassOf _:bnode)，而 _:bnode 上写了 rdfs:comment
        if promote_anonymous:
            # 找到所有匿名主体上写的注释
            for anon in (s for s in subjects_with_comment if isinstance(s, BNode)):
                comments = list(g.objects(anon, RDFS.comment))
                if not comments:
                    continue

                # 找引用这个匿名节点当“父类限制”的命名类： X rdfs:subClassOf anon
                targets = [x for x in g.subjects(RDFS.subClassOf, anon) if isinstance(x, URIRef) and _keep(x)]

                # 若找不到“引用者”，再尝试把匿名节点当作“子类”，向上找它的命名父类： anon rdfs:subClassOf Y
                if not targets:
                    targets = [y for y in g.objects(anon, RDFS.subClassOf) if isinstance(y, URIRef) and _keep(y)]

                # 仍找不到就给它一个临时命名（尽量减少）：如果你坚决不要任何“Anon_”占位，可注释掉下面 4 行
                if not targets:
                    tmp_local = f"AnonPromoted_{_local_name(onto_name)}"
                    for cm in comments:
                        triples_out.add((tmp_local, 'comment', str(cm)))
                    continue

                # 将匿名注释“归属”到这些命名类（可限制数量）
                for tgt in targets[:promote_max_targets]:
                    cls_name = _local_name(tgt)
                    for cm in comments:
                        triples_out.add((cls_name, 'comment', str(cm)))

        # 3) 写表
        df = pd.DataFrame(sorted(triples_out), columns=['Class', 'Relationship', 'Comment'])
        df.to_excel(output_path, index=False)
        return triples_out

    def get_class_definition_triples(self, onto_name, results_folder=save_folder + '/definition'):
        """
        Retrieves all class definitions (including from imported ontologies) and stores them as triples:
        (Class, 'definition', DefinitionValue).
        """
        class_definitions_set = set()
        processed_classes = set()  # 防止重复处理类

        # 支持多种可能的 definition 属性名
        definition_props = ['definition', 'IAO_0000115']

        def get_definitions(cls):
            for prop in definition_props:
                if hasattr(cls, prop):
                    value = getattr(cls, prop)
                    if value:
                        return value
            return []

        def get_fallback_textual_comment(cls):
            for prop in cls.namespace.ontology.annotation_properties():
                if prop and hasattr(prop, "name") and 'term_tracker_annotation' in prop.name.lower():
                    try:
                        values = prop[cls]
                        for value in values:
                            if isinstance(value, str) and not value.startswith('http'):
                                return value.strip()
                    except:
                        continue
            return None

        def add_definitions_from_ontology(ontology):
            for cls in ontology.classes():
                if cls in processed_classes or cls.name == "Thing":
                    continue
                processed_classes.add(cls)

                class_name = self._get_label_or_name(cls)
                definitions = get_definitions(cls)
                if definitions:
                    for definition in definitions:
                        class_definitions_set.add((class_name, 'definition', str(definition).strip()))
                else:
                    fallback = get_fallback_textual_comment(cls)
                    if fallback:
                        class_definitions_set.add((class_name, 'definition', fallback))
                    else:
                        class_definitions_set.add((class_name, 'definition', 'None'))

        onto = self.onto()
        add_definitions_from_ontology(onto)

        for imported_onto in onto.imported_ontologies:
            if not imported_onto.loaded:
                imported_onto.load()
            add_definitions_from_ontology(imported_onto)

        print(f"Definitions from {self.name} loaded successfully!")

        # Save file
        df = pd.DataFrame(class_definitions_set, columns=['Class', 'Relationship', 'Definition'])
        suffix = '_ontology_class_definition_triples.xlsx'
        output_path = get_file_path(results_folder, onto_name, suffix)
        df.to_excel(output_path, index=False)

        return class_definitions_set

    def get_data_property_comment_triples(self, onto_name, results_folder=save_folder + '/comment'):
        data_property_comments_set = set()
        # Iterate over each data property in the ontology
        for prop in self.onto().data_properties():
            prop_name = prop.name
            # Retrieve and store each property´s comments
            for comment in prop.comment:
                data_property_comments_set.add((prop_name, 'comment', str(comment)))
                if not any(prop.comment):
                    data_property_comments_set.add((prop_name, 'comment', 'None'))

        df = pd.DataFrame(data_property_comments_set, columns=['Data Property', 'Relationship', 'Comment'])
        suffix = '_ontology_data_property_comment_triples.xlsx'
        output_path = get_file_path(results_folder, onto_name, suffix)
        df.to_excel(output_path, index=False)
        return data_property_comments_set

    def get_object_property_comment_triples(self, onto_name, results_folder=save_folder + '/comment'):
        object_property_set = set()
        object_property_comments_set = set()
        # Iterate over each object property in the ontology
        for prop in self.onto().object_properties():
            prop_name = prop.name
            # Retrieve and store each property´s comments
            for comment in prop.comment:
                object_property_comments_set.add((prop_name, 'comment', str(comment)))
                if not any(prop.comment):
                    object_property_comments_set.add((prop_name, 'comment', 'None'))
            object_property_set.add(prop_name)

        df = pd.DataFrame(object_property_comments_set, columns=['Object Property', 'Relationship', 'Comment'])
        suffix = '_ontology_object_property_comment_triples.xlsx'
        output_path = get_file_path(results_folder, onto_name, suffix)
        df.to_excel(output_path, index=False)
        print(f'Comments of object properties from {self.name} loaded successfully!')
        return object_property_set, object_property_comments_set

    def get_property_comment(self, data_type):
        data_comments_set = set()
        if data_type == 'data_property':
            for prop in self.onto().data_properties():
                prop_name = prop.name
                for comment in prop.comment:
                    data_comments_set.add((prop_name, str(comment)))
                    if not any(prop.comment):
                        data_comments_set.add((prop_name, 'None'))
        elif data_type == 'object_property':
            for prop in self.onto().object_properties():
                prop_name = prop.name
                for comment in prop.comment:
                    data_comments_set.add((prop_name, str(comment)))
                    if not any(prop.comment):
                        data_comments_set.add((prop_name, 'None'))
        data_comments_list = list(data_comments_set)
        return data_comments_list

    def get_property_structure_triple(self, data_type):
        if data_type not in ("data_property", "object_property"):
            raise ValueError("data_type must be 'data_property' or 'object_property'")

        property_structure_set = set()

        props_iter = (
            self.onto().data_properties()
            if data_type == "data_property"
            else self.onto().object_properties()
        )

        for prop in props_iter:
            prop_name = getattr(prop, "name", str(prop))

            try:
                domains = list(prop.domain)
            except Exception:
                domains = []
            if not domains:
                domains = [None]

            try:
                ranges = list(prop.range)
            except Exception:
                ranges = []
            if not ranges:
                ranges = [None]

            for dom in domains:
                domain_name = dom.name if dom and hasattr(dom, "name") else "None"
                for rng in ranges:
                    if rng is None:
                        range_name = "None"
                    else:
                        range_name = (
                            rng.name
                            if hasattr(rng, "name")
                            else str(rng).replace("<class '", "").replace("'>", "")
                        )
                    property_structure_set.add((domain_name, prop_name, range_name))

        return list(property_structure_set)

    def save_data(self, onto_name, suffix, data, columns):
        output_dir = "../Data/onto_data"
        file_name = onto_name
        output_path = get_file_path(output_dir, file_name, suffix)

        df = pd.DataFrame(data, columns=[columns])
        df.to_excel(output_path, index=False)


if __name__ == '__main__':
    onto = OntologyReading(r'..\energy_domain_experiment/ontology/Sargon.owl')
    onto.get_classes()
