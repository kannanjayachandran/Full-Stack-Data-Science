# Style Guide

These are some general styling (syntactic and semantic) guidelines specific to this project. The purpose of such a style guide is to ensure quality of the content and uniformity.

## Coding 

Python : PEP8 standards

## Jupyter Notebooks

- Notebooks are meant to supplement the notes (in readme files) or to showcase implementation of the concept, avoid adding large amount of theories or notes to it.

- Name them appropriately and add them in the `notebooks` folder in the respective folder.

## Markdown Style 

Using `html` in markdown is generally a bad idea, and should be avoided as far as possible, but for certain styling purposes described below it may be used.

1. Centering and coloring the heading

```markdown 
<h1 align="center" style="color: orange">Content</h1>
```

> Use only `Orange` and `light grey` for heading 
 
2. Centering image

```markdown
<div align="center">

![]()

</div>
```

For mathematical equations using latex, inline equations should be colored using Orange (#F99417). Block equations should not be colored.