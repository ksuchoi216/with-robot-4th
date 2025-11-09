# Robot Arm API Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Basic Movement Functions](#examplar-basic-movement-functions)
3. [Object Interactions](#examplar-object-interactions)
4. [Environment-Specific Functions](#examplar-environment-specific-functions)
5. [High-Level Tasks](#examplar-high-level-tasks)
6. [Task Resolution Process](#task-resolution-Process)
7. [Parameters and Safety Information](#parameters-and-safety-information)

<a name="introduction"></a>
## Introduction
This document provides detailed instructions for controlling the PandaOmron robot. The code snippets included here can be executed directly without additional setup. Example:

````python
set_target_position(0, 0, PI)
set_target_position(-0.5, 0, PI)
````
