# Style Guide

These are some general styling guidelines specific to this project. The purpose of such a style guide is to ensure quality of the content and uniformity.

## Coding 

Python : PEP8 standards

## Jupyter Notebooks

- Notebooks are meant to supplement the notes present in readme files or to showcase implementation of the concept only. They **should not** be used as a replacement for readme files.

- Notebooks should be present inside the `notebooks` directory inside the respective topic directory.

## Markdown Style 

Using `html` in markdown is generally a bad idea, and should be avoided as far as possible, but for certain styling purposes described below it may be used (aesthetics purposes only).

1. Centering and coloring the heading

```markdown 
<h1 align="center" style="color: orange">Most of the headings</h1>
```
 
2. Centering image

```markdown
<div align="center">

![]()

</div>
```

## LaTeX Style
- For mathematical equations using latex, inline equations should be colored using `Orange (#F99417)`. Block equations should not be colored.

- Github flavoured markdown does not support latex completely, so currently the best way to include latex is to use `html` tags.
