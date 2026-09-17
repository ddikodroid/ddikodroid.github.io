---
layout: default
title: Block Leave Journal
permalink: /block-leave/
---

# Block Leave Journal

A running log of activities during block leave.

<ul>
{% for post in site.categories['block-leave'] %}
  <li>
    <strong>{{ post.date | date: "%B %-d, %Y" }}</strong> —
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </li>
{% endfor %}
</ul>
