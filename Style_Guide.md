# Style Guide

As the project grew, I realized two things large changes are extremely difficult to implement and lack of uniformity (some structure) in the content. This style guide attempts to address the both of these issues.

## Coding 

**Python** : PEP8 standards

## Jupyter Notebooks

- Jupyter notebooks are great interactive tools, but here they are meant to supplement the notes (markdown) or to showcase implementation of the concept only. Avoid writing notes in the notebook.

- All notebooks should be present inside the `notebooks` directory inside the respective topic directory. If they are the only content in the topic, they can be placed in the respective topic directory.

## Markdown Style 

Using `html` in markdown is generally a bad idea, and should be avoided as far as possible, but for certain styling purposes (_aesthetics_) described below it may be used.

1. Centering the content

```markdown 
<h1 align="center">Main Readme Heading</h1>
```
 
2. Centering image

```markdown
<p align="center">
    <a href="#heading">
        <img src="./img/path.png" alt="Logo" height=380>
    </a>
</p>
```

## LaTeX Style

- For mathematical equations using latex, inline equations should be colored using `Orange (#F99417)`. Block equations should not be colored.
