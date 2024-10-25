import warnings
from typing import Callable, Sequence

from src.retrieval.models import Document, DocumentChunk
from src.retrieval.utils import length_fn
from tree_sitter_languages import get_parser

warnings.filterwarnings("ignore")


def pretty_print_node(node, indent=0):
    print("  " * indent + f"{node.type} [{node.start_point}, {node.end_point}]")
    for child in node.children:
        pretty_print_node(child, indent + 1)


def collect_imports(node):
    imports = []
    for child in node.children:
        if child.type in ("import_statement", "import_from_statement"):
            imports.append(child)
    return imports


def build_import_map(import_nodes):
    import_map = {}
    for imp in import_nodes:
        imp_text = imp.text.decode("utf-8")
        if imp.type == "import_from_statement":
            for child in imp.children[3:]:
                if child.type == "dotted_name":
                    name = child.text.decode("utf-8")
                    import_map[name] = imp_text
                elif child.type == "aliased_import":
                    alias = child.children[2].text.decode("utf-8")
                    import_map[alias] = imp_text
        elif imp.type == "import_statement":
            for child in imp.children:
                if child.type == "dotted_name":
                    name = child.text.decode("utf-8")
                    import_map[name] = imp_text
                elif child.type == "aliased_import":
                    alias = child.children[2].text.decode("utf-8")
                    import_map[alias] = imp_text
    return import_map


def collect_module_variables(node):
    module_variables = []
    for child in node.children:
        if child.type == "expression_statement":
            assignment = child.children[0]
            if assignment.type == "assignment":
                target = assignment.child_by_field_name("left")
                if target and target.type == "identifier":
                    var_name = target.text.decode("utf-8")
                    module_variables.append(var_name)
    return module_variables


def collect_identifiers(node, identifiers):
    if node.type == "identifier":
        identifiers.add(node.text.decode("utf-8"))
    elif node.type in ("function_definition", "class_definition"):
        parameters = node.child_by_field_name("parameters")
        if parameters:
            collect_identifiers(parameters, identifiers)
        return_type = node.child_by_field_name("return_type")
        if return_type:
            collect_identifiers(return_type, identifiers)
    else:
        for child in node.children:
            collect_identifiers(child, identifiers)


def collect_identifiers_from_type_annotations(node, identifiers):
    if node.type in ("type", "type_identifier", "identifier"):
        identifiers.add(node.text.decode("utf-8"))
    else:
        for child in node.children:
            collect_identifiers_from_type_annotations(child, identifiers)


def process_decorated_definition(node, context, import_map, module_variables):
    chunks = []
    identifiers = set()
    collect_identifiers(node, identifiers)
    func_def = None
    for child in node.children:
        if child.type in ("function_definition", "class_definition"):
            func_def = child
            break

    if func_def:
        func_name_node = func_def.child_by_field_name("name")
        func_name = func_name_node.text.decode("utf-8")
        func_code = node.text.decode("utf-8")

        used_imports = []
        used_module_vars = []

        for ident in identifiers:
            if ident in import_map:
                used_imports.append(import_map[ident])
            if ident in module_variables:
                used_module_vars.append(ident)

        func_context = {}

        if used_imports:
            func_context["imports"] = list(set(used_imports))

        if used_module_vars:
            func_context["module_variables"] = used_module_vars

        if "parent_class" in context:
            func_context["parent_class"] = context["parent_class"]

        if "parent_function" in context:
            func_context["parent_function"] = context["parent_function"]

        if not func_context:
            func_context = None

        result = {
            "type": "function" if func_def.type == "function_definition" else "class",
            "name": func_name,
            "definition": func_code,
            "context": func_context,
        }

        if "parent_class" in context and func_def.type == "function_definition":
            result["type"] = "method"

        chunks.append(result)
    return chunks


def process_function(node, context, import_map, module_variables):
    chunks = []
    func_name_node = node.child_by_field_name("name")
    func_name = func_name_node.text.decode("utf-8")

    func_code = node.text.decode("utf-8")

    identifiers = set()
    func_body = node.child_by_field_name("body")
    collect_identifiers(func_body, identifiers)

    parameters = node.child_by_field_name("parameters")
    if parameters:
        collect_identifiers_from_type_annotations(parameters, identifiers)
    return_type = node.child_by_field_name("return_type")
    if return_type:
        collect_identifiers_from_type_annotations(return_type, identifiers)

    used_imports = []
    used_module_vars = []

    for ident in identifiers:
        if ident in import_map:
            used_imports.append(import_map[ident])
        if ident in module_variables:
            used_module_vars.append(ident)

    func_context = {}

    if used_imports:
        func_context["imports"] = list(set(used_imports))

    if used_module_vars:
        func_context["module_variables"] = used_module_vars

    if "parent_class" in context:
        func_context["parent_class"] = context["parent_class"]

    if "parent_function" in context:
        func_context["parent_function"] = context["parent_function"]

    if not func_context:
        func_context = None

    result = {
        "type": "function",
        "name": func_name,
        "definition": func_code,
        "context": func_context,
    }

    if "parent_class" in context:
        result["type"] = "method"

    chunks.append(result)

    # Process any nested functions
    for child in func_body.children:
        if child.type == "function_definition":
            context_copy = context.copy()
            context_copy["parent_function"] = func_name
            chunks.extend(
                process_function(child, context_copy, import_map, module_variables)
            )
        elif child.type == "decorated_definition":
            chunks.extend(
                process_decorated_definition(
                    child, context, import_map, module_variables
                )
            )
    return chunks


def process_class(node, context, import_map, module_variables):
    chunks = []
    class_name_node = node.child_by_field_name("name")
    class_name = class_name_node.text.decode("utf-8")

    class_body = node.child_by_field_name("body")

    class_attributes = []
    init_method = None
    other_methods = []
    identifiers = set()

    for child in class_body.children:
        if child.type == "expression_statement":
            assignment = child.children[0]
            if assignment.type == "assignment":
                class_attributes.append(assignment.text.decode("utf-8"))
                collect_identifiers(assignment, identifiers)
            elif assignment.type == "typed_parameter":
                class_attributes.append(assignment.text.decode("utf-8"))
                collect_identifiers_from_type_annotations(assignment, identifiers)
        elif child.type == "function_definition":
            func_name_node = child.child_by_field_name("name")
            func_name = func_name_node.text.decode("utf-8")
            if func_name == "__init__":
                init_method = child
            else:
                other_methods.append(child)
        elif child.type == "decorated_definition":
            func_def = None
            for c in child.children:
                if c.type == "function_definition":
                    func_def = c
                    break
            if func_def:
                func_name_node = func_def.child_by_field_name("name")
                func_name = func_name_node.text.decode("utf-8")
                if func_name == "__init__":
                    init_method = child
                else:
                    other_methods.append(child)
    used_imports = []
    used_module_vars = []

    for ident in identifiers:
        if ident in import_map:
            used_imports.append(import_map[ident])
        if ident in module_variables:
            used_module_vars.append(ident)

    # Build the context
    class_context = {}

    if used_imports:
        class_context["imports"] = list(set(used_imports))

    if used_module_vars:
        class_context["module_variables"] = used_module_vars

    if not class_context:
        class_context = None

    class_def_lines = ["class " + class_name + ":"]

    for attr in class_attributes:
        class_def_lines.append("    " + attr)

    if init_method:
        init_code = "\n\n    " + init_method.text.decode("utf-8")
        init_lines = init_code.split("\n")
        for line in init_lines:
            class_def_lines.append(line)

    class_definition = "\n".join(class_def_lines)

    chunk = {
        "type": "class",
        "name": class_name,
        "definition": class_definition,
        "context": class_context,
    }

    chunks.append(chunk)

    for method in other_methods:
        context_copy = context.copy()
        context_copy["parent_class"] = class_name
        if method.type == "decorated_definition":
            chunks.extend(
                process_decorated_definition(
                    method, context_copy, import_map, module_variables
                )
            )
        else:
            chunks.extend(
                process_function(method, context_copy, import_map, module_variables)
            )
    return chunks


def process_root(node, import_map, module_variables):
    chunks = []
    module_code_lines = []
    module_imports = []
    for child in node.children:
        if child.type in ("import_statement", "import_from_statement"):
            import_text = child.text.decode("utf-8")
            module_code_lines.append(import_text)
            module_imports.append(import_text)
        elif child.type == "class_definition":
            chunks.extend(
                process_class(
                    child,
                    context={},
                    import_map=import_map,
                    module_variables=module_variables,
                )
            )
        elif child.type == "function_definition":
            chunks.extend(
                process_function(
                    child,
                    context={},
                    import_map=import_map,
                    module_variables=module_variables,
                )
            )
        elif child.type == "decorated_definition":
            chunks.extend(
                process_decorated_definition(
                    child,
                    context={},
                    import_map=import_map,
                    module_variables=module_variables,
                )
            )
    return chunks


def convert_chunks_to_strs(chunks):
    str_chunks = []
    for chunk in chunks:
        chunk_str = ""
        indented = False
        chunk_context = chunk["context"]
        if chunk_context is not None and chunk["type"] != "module":
            chunk_context_str = ""
            if "imports" in chunk_context:
                chunk_context_str += "\n".join(chunk_context["imports"]) + "\n"
            if "module_variables" in chunk_context:
                chunk_context_str += "\n".join(chunk_context["module_variables"]) + "\n"
            if "parent_class" in chunk_context:
                chunk_context_str += (
                    f"\nclass: {chunk_context['parent_class']}" + "\n    # ... (more)\n"
                )
                indented = True
            if "parent_function" in chunk_context:
                chunk_context_str += (
                    f"\ndef: {chunk_context['parent_function']}"
                    + "\n    # ... ( (more)\n"
                )
                indented = True
            chunk_str += f"{chunk_context_str}" + "\n    "
        if indented:
            chunk_str += "    "
        else:
            chunk_str = chunk_str[:-4]

        chunk_str += f"{chunk['definition']}"
        str_chunks.append(chunk_str)
    return str_chunks


def chunk_code(
    doc: Document, length_fn: Callable[[str], int]
) -> Sequence[DocumentChunk]:
    document_dict = doc.model_dump(mode="json")
    parser = get_parser("python")
    code = doc.content.encode("utf-8")
    tree = parser.parse(code)
    import_nodes = collect_imports(tree.root_node)
    import_map = build_import_map(import_nodes)
    module_variables = collect_module_variables(tree.root_node)
    chunks = process_root(tree.root_node, import_map, module_variables)
    chunks_strs = convert_chunks_to_strs(chunks)
    chunks_strs = list(filter(lambda x: len(x.strip().splitlines()) > 1, chunks_strs))
    chunks_info = [
        {
            "embed_content": chunk_str,
            "content": chunk_str,
            "embed_tokens": length_fn(chunk_str),
            "num_tokens": length_fn(chunk_str),
        }
        for chunk_str in chunks_strs
    ]
    return [
        DocumentChunk(**{"document_id": doc.id, **document_dict, **chunk})
        for chunk in chunks_info
    ]
