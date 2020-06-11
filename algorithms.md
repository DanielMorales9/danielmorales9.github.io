---
layout: page
title: Kata Algorithms
permalink: /algorithms/
---
{% for algo in site.algorithms %}
<h2>
<a href="{{ algo.url }}">
  {{ algo.name }}
</a>
</h2>
{% endfor %}