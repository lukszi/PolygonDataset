#!/usr/bin/env python3
"""Build markdown documentation for polygon-dataset package using pydoc-markdown API."""

import os
import re
import sys
from typing import List, Dict, Any, Optional, Set
import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate documentation for polygon-dataset package using pydoc-markdown API.")
    parser.add_argument("--output-dir", "-o", default="docs",
                        help="Output directory for documentation (default: docs)")
    parser.add_argument("--pyproject", "-p", default="pyproject.toml",
                        help="Path to pyproject.toml file (default: pyproject.toml)")
    parser.add_argument("--source-dir", "-s", default="src",
                        help="Source directory (default: src)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    return parser.parse_args()


def install_if_missing(package: str, verbose: bool = False) -> None:
    """Install a package if it's not already installed."""
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        if verbose:
            print(f"Installing {package}...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", package],
                       check=True,
                       stdout=subprocess.DEVNULL if not verbose else None,
                       stderr=subprocess.DEVNULL if not verbose else None)


def read_packages_from_pyproject(pyproject_path: str, verbose: bool = False) -> List[str]:
    """Read the list of packages from pyproject.toml file."""
    # Ensure we have a TOML parser
    install_if_missing("tomli", verbose)
    import tomli

    try:
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomli.load(f)

        # Look for packages in tool.setuptools.packages
        packages = pyproject_data.get("tool", {}).get("setuptools", {}).get("packages", [])

        if not packages:
            raise ValueError(f"No packages found in {pyproject_path} under tool.setuptools.packages")

        return packages
    except FileNotFoundError:
        raise FileNotFoundError(f"pyproject.toml file not found at {pyproject_path}")
    except Exception as e:
        raise ValueError(f"Error reading pyproject.toml: {e}")


def clean_documentation(content: str, verbose: bool = False) -> str:
    """Clean the documentation content of unwanted elements."""
    # Pattern to match HTML anchor tags
    anchor_pattern = r'<a id="[^"]+"></a>\n?'

    # Pattern to match unnecessary import headers like '## Path'
    import_pattern = r'\n## (Path|Optional|Union|List|Dict|Any|Set|Type|TypeVar|Tuple|Callable|ClassVar)\n'

    # Pattern for __all__ sections with no content
    all_pattern = r'\n#### .*\.__all__.*?\n\n'

    # Count original occurrences for reporting
    orig_anchor_count = len(re.findall(anchor_pattern, content))
    orig_import_count = len(re.findall(import_pattern, content))
    orig_all_count = len(re.findall(all_pattern, content))

    # Remove the unwanted elements
    cleaned_content = re.sub(anchor_pattern, '', content)
    cleaned_content = re.sub(import_pattern, '', cleaned_content)
    cleaned_content = re.sub(all_pattern, '\n', cleaned_content)

    # Clean up empty sections (header followed by another header or end of text)
    cleaned_content = re.sub(r'(## \w+)\n+(?=## |\Z)', '', cleaned_content)

    if verbose:
        print(
            f"Removed: {orig_anchor_count} HTML anchors, {orig_import_count} import headers, {orig_all_count} __all__ sections")

    return cleaned_content


def get_module_hierarchy(modules: List[Any]) -> Dict[str, Dict]:
    """Create a hierarchy of modules for organizing documentation files."""
    hierarchy = {}

    for module in modules:
        # Skip modules that don't have any documented members
        if not has_documented_members(module):
            continue

        parts = module.name.split('.')
        current = hierarchy

        # Build the hierarchy
        for i, part in enumerate(parts):
            if part not in current:
                current[part] = {"__module__": None, "__children__": {}}

            # If this is the last part, store the module
            if i == len(parts) - 1:
                current[part]["__module__"] = module

            current = current[part]["__children__"]

    return hierarchy


def has_documented_members(module) -> bool:
    """Check if a module has any documented members."""
    # Check if the module itself has a docstring
    if module.docstring:
        return True

    # Check if any of its members have docstrings
    for member in module.members:
        if member.docstring:
            return True

    return False


def render_module_documentation(
        renderer: Any,
        module: Any,
        output_dir: str,
        file_path: List[str],
        verbose: bool = False
) -> None:
    """Render documentation for a single module to a file."""
    if not module:
        return

    # Create the module documentation
    content = renderer.render_to_string([module])

    # Clean the content
    cleaned_content = clean_documentation(content, verbose)

    # Skip empty documentation
    if not cleaned_content.strip():
        if verbose:
            print(f"Skipping empty documentation for {module.name}")
        return

    # Create file path
    rel_path = os.path.join(*file_path) + ".md"
    full_path = os.path.join(output_dir, rel_path)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # Write the documentation
    with open(full_path, "w") as f:
        f.write(cleaned_content)

    if verbose:
        print(f"Created documentation file: {rel_path}")


def render_package_index(
        name: str,
        children: Dict[str, Dict],
        output_dir: str,
        file_path: List[str],
        verbose: bool = False
) -> None:
    """Render an index file for a package with links to its modules."""
    # Create the index content
    content = f"# {name}\n\n"

    # Get module documentation if this package has a module
    module = children.get("__module__")
    if module and module.docstring:
        content += f"{module.docstring}\n\n"

    # Add module link only if it has content
    if module and has_documented_members(module):
        module_name = module.name.split(".")[-1]
        content += f"## Modules\n\n"
        content += f"- [{module_name}](./{module_name}.md)\n\n"

    # Add links to child packages/modules
    child_packages = []
    sorted_children = sorted(children.get("__children__", {}).keys())
    for child in sorted_children:
        child_info = children["__children__"][child]
        # Only include packages with content
        if child_info.get("__module__") or child_info.get("__children__"):
            child_packages.append(child)

    if child_packages:
        content += f"## Subpackages\n\n"
        for child in child_packages:
            content += f"- [{child}](./{child}/index.md)\n"

    # Skip empty index files
    if content.strip() == f"# {name}":
        if verbose:
            print(f"Skipping empty index for {name}")
        return

    # Create file path
    rel_path = os.path.join(*file_path, "index.md")
    full_path = os.path.join(output_dir, rel_path)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # Write the index
    with open(full_path, "w") as f:
        f.write(content)

    if verbose:
        print(f"Created index file: {rel_path}")


def process_hierarchy(
        renderer: Any,
        hierarchy: Dict[str, Dict],
        output_dir: str,
        parent_path: List[str] = None,
        verbose: bool = False
) -> None:
    """Process the module hierarchy and create documentation files."""
    if parent_path is None:
        parent_path = []

    for name, info in hierarchy.items():
        current_path = parent_path + [name]
        module = info.get("__module__")
        children = info.get("__children__", {})

        # Create an index file for this package
        render_package_index(name, info, output_dir, current_path, verbose)

        # Render the module's documentation if it exists
        if module and has_documented_members(module):
            render_module_documentation(renderer, module, output_dir, current_path, verbose)

        # Process children recursively
        if children:
            process_hierarchy(renderer, children, output_dir, current_path, verbose)


def generate_documentation(
        package_paths: List[str],
        output_dir: str,
        source_dir: str = "src",
        verbose: bool = False
) -> None:
    """Generate documentation using the pydoc-markdown API."""
    # Ensure pydoc-markdown is installed
    install_if_missing("pydoc-markdown", verbose)

    # Import pydoc-markdown modules
    from pydoc_markdown.interfaces import Context, Processor
    from pydoc_markdown.contrib.loaders.python import PythonLoader
    from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer
    from pydoc_markdown.contrib.processors.filter import FilterProcessor
    from pydoc_markdown.contrib.processors.smart import SmartProcessor
    from pydoc_markdown.util.docspec import Resolver

    # Initialize pydoc-markdown context
    context = Context(directory='.')

    # Use absolute paths for search_path
    abs_source_dir = os.path.abspath(source_dir)
    if verbose:
        print(f"Using source directory: {abs_source_dir}")

    # Create the loader with filter for unwanted packages
    loader = PythonLoader(
        search_path=[abs_source_dir],
        modules=[]  # Will be filled dynamically
    )

    # Create processors to improve documentation quality
    filter_processor = FilterProcessor(
        documented_only=True,  # Skip undocumented members
        exclude_private=True,  # Skip private members
        exclude_special=True  # Skip special methods (__xxx__)
    )

    smart_processor = SmartProcessor()

    # Create the renderer with better settings
    renderer = MarkdownRenderer(
        render_module_header=True,
        descriptive_class_title=True,
        add_method_class_prefix=True,
        add_member_class_prefix=True,
        classdef_code_block=True,
        render_toc=False  # Disable table of contents in each file
    )

    # Build a list of modules to process based on package paths
    module_names = []
    for package_path in package_paths:
        relative_path = os.path.relpath(package_path, source_dir)
        package_name = relative_path.replace(os.path.sep, ".")

        # Add only the main package
        module_names.append(package_name)

    if verbose:
        print(f"Using packages to document: {module_names}")

    # Set the modules to load
    loader.modules = module_names

    # Initialize components
    loader.init(context)
    filter_processor.init(context)
    smart_processor.init(context)
    renderer.init(context)

    # Load all modules
    if verbose:
        print("Loading modules...")
    modules = list(loader.load())  # Convert generator to list

    if not modules:
        raise ValueError("No modules were loaded. Check your source directory and package paths.")

    if verbose:
        print(f"Loaded {len(modules)} modules")

    # Filter out unwanted standard library modules
    excluded_modules = ['typing', 'types', 'enum', 'abc', 're', 'os', 'sys', 'pathlib', 'collections']
    original_count = len(modules)
    modules = [m for m in modules if not any(m.name.startswith(exclude) for exclude in excluded_modules)]

    if verbose and original_count != len(modules):
        print(f"Filtered out {original_count - len(modules)} standard library modules")

    # Process the modules to improve their structure
    resolver = Resolver(modules)
    modules = filter_processor.process(modules, resolver)
    modules = smart_processor.process(modules, resolver)

    if verbose:
        print(f"After processing: {len(modules)} modules remain")

    # Create a hierarchy of modules
    hierarchy = get_module_hierarchy(modules)

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process the hierarchy and create documentation files
    process_hierarchy(renderer, hierarchy, output_dir, verbose=verbose)

    # Create main index file
    main_index_content = "# Polygon Dataset Documentation\n\n"

    # Add top-level packages
    if hierarchy:
        main_index_content += "## Packages\n\n"
        for package in sorted(hierarchy.keys()):
            # Only include packages that have content
            if os.path.exists(os.path.join(output_dir, package, "index.md")):
                main_index_content += f"- [{package}](./{package}/index.md)\n"

    with open(os.path.join(output_dir, "index.md"), "w") as f:
        f.write(main_index_content)

    if verbose:
        print(f"Documentation generated in {output_dir}")


def main() -> None:
    """Build the documentation."""
    args = parse_args()

    try:
        # Read packages from pyproject.toml
        packages = read_packages_from_pyproject(args.pyproject, args.verbose)

        if args.verbose:
            print(f"Found packages to document: {packages}")

        # Convert packages to paths
        package_paths = []
        for package in packages:
            path = os.path.join(args.source_dir, package.replace(".", os.path.sep))
            if os.path.exists(path):
                package_paths.append(path)
                if args.verbose:
                    print(f"Found package path: {path}")
            else:
                print(f"Warning: Package path not found: {path}")

        if not package_paths:
            raise ValueError("No valid package paths found. Check your pyproject.toml and source directory.")

        # Generate the documentation
        generate_documentation(
            package_paths=package_paths,
            output_dir=args.output_dir,
            source_dir=args.source_dir,
            verbose=args.verbose
        )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()