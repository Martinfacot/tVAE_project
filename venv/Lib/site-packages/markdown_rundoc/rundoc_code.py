"""
Rundoc Code Extension for Python Markdown
=========================================

This extension adds Rundoc Code Blocks to Python-Markdown.

Based on Fenced Code Extension:
    See <https://Python-Markdown.github.io/extensions/rundoc_code_blocks>
    for documentation.
    Original code Copyright 2007-2008 [Waylan Limberg](http://achinghead.com/).
    Changes Copyright 2008-2014 The Python Markdown Project

Changes Copyright 2018-2019 Predrag Mandic

License: [BSD](http://www.opensource.org/licenses/bsd-license.php)
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
import re

env_tags = [ 'env', 'environ', 'environment', 'secret', 'secrets' ]

class RundocCodeExtension(Extension):

    def __init__(self, **kwargs):
        self.config = {
            'tags': [ kwargs.get('tags') ],
            'must_have_tags': [ kwargs.get('must_have_tags') ],
            'must_not_have_tags': [ kwargs.get('must_not_have_tags') ],
            'single_session': [ kwargs.get('single_session') ],
            'selection_tag': [ kwargs.get('selection_tag') ],
            }
        super(RundocCodeExtension, self).__init__(**kwargs)
    
    def extendMarkdown(self, md, md_globals):
        """ Add RundocBlockPreprocessor to the Markdown instance. """
        md.registerExtension(self)

        md.preprocessors.add('rundoc_code_block',
                             RundocBlockPreprocessor(md),
                             ">normalize_whitespace")

def is_selected(
    code_match, #SRE_Match object
    have_tags,
    must_have_tags,
    must_not_have_tags,
    single_session='',
    collected=[], # select envs based on these
    skip=[] # skip selection on these interpreters
    ):
    """ Return True if regex SRE_Match object code_match is selected. """
    tags = []
    have_tags.extend(must_have_tags)
    if code_match.group('tags'):
        tags = code_match.group('tags').split('#')
        tags = list(filter(bool, tags))
    if not len(tags): return False # no interpreter
    if tags[0] in skip: return False
    if tags[0] in env_tags:
        for tag in tags:
            if tag in collected:
                return True # collected env
    elif single_session and single_session != tags[0]:
        return False
    selected = False
    if have_tags:
        for tag in have_tags:
            if tag in tags:
                selected = True
                break
    if must_have_tags:
        for tag in must_have_tags:
            if tag not in tags:
                selected = False
                break
    if selected and must_not_have_tags:
        for tag in must_not_have_tags:
            if tag in tags:
                selected = False
                break
    if not (have_tags or must_have_tags or must_not_have_tags):
        if len(tags):
            selected = True
    return selected

class RundocBlockPreprocessor(Preprocessor):
    RUNDOC_BLOCK_RE = re.compile(r'''
(?P<fence>^(?:~{3,}|`{3,}))[ ]*         # Opening ``` or ~~~
(\{?\.?(?P<tags>[^\n\r]*))?[ ]*         # Optional {, and lang
# Optional highlight lines, single- or double-quote-delimited
(hl_lines=(?P<quot>"|')(?P<hl_lines>.*?)(?P=quot))?[ ]*
}?[ ]*\n                                # Optional closing }
(?P<code>.*?)(?<=\n)
(?P=fence)[ ]*$''', re.MULTILINE | re.DOTALL | re.VERBOSE)
    config = None

    def __init__(self, md):
        super(RundocBlockPreprocessor, self).__init__(md)

    def run(self, lines):
        """ Match and store Rundoc Code Blocks in the HtmlStash. """

        # Get config
        for ext in self.markdown.registeredExtensions:
            if isinstance(ext, RundocCodeExtension):
                self.config = ext.config
                break
        have_tags = []
        must_have_tags = []
        must_not_have_tags = []
        single_session = ''
        collected_tags = set() # all tags of all selected code blocks
        if self.config['tags'][0]:
            have_tags = self.config['tags'][0].split('#')
            have_tags = list(filter(bool, have_tags))
        if self.config['must_have_tags'][0]:
            must_have_tags = self.config['must_have_tags'][0].split('#')
            must_have_tags = list(filter(bool, must_have_tags))
        if self.config['must_not_have_tags'][0]:
            must_not_have_tags = self.config['must_not_have_tags'][0].split('#')
            must_not_have_tags = list(filter(bool, must_not_have_tags))
        if self.config['single_session'][0]:
            single_session = self.config['single_session'][0]

        # Iterate over text. Find first code block, collect all of it's tags if
        # identified as selected and is not env or secret. Remove it and start
        # again until no code blocks are left.
        text = "\n".join(lines)
        while True:
            m = self.RUNDOC_BLOCK_RE.search(text)
            if not m: break
            selected_non_env = is_selected(
                m,
                have_tags,
                must_have_tags,
                must_not_have_tags,
                single_session,
                collected=[],
                skip=env_tags )
            if selected_non_env:
                tags = m.group('tags').split('#')
                collected_tags.update(tags[1:])
            text = text[m.end():]

        # Iterate over text. Find first code block. Determine html classes
        # based on tags and selected state. Replace it with html. Start over
        # until no code blocks are left.
        text = "\n".join(lines)
        while True:
            m = self.RUNDOC_BLOCK_RE.search(text)
            if not m: break
            tags = m.group('tags').split('#')
            classes = list(filter(bool, tags))
            selected = is_selected(
                m,
                have_tags,
                must_have_tags,
                must_not_have_tags,
                single_session,
                collected=list(collected_tags))
            if selected:
                classes.append(self.config['selection_tag'][0] or 'selected')

            class_tag = ' class="%s"' % ' '.join(classes)
            code = '<pre><code%s>%s</code></pre>' % (class_tag,
                                     self._escape(m.group('code')))

            placeholder = self.markdown.htmlStash.store(code, safe=True)
            text = '%s\n%s\n%s' % (text[:m.start()],
                                   placeholder,
                                   text[m.end():])
        return text.split("\n")

    def _escape(self, txt):
        """ basic html escaping """
        txt = txt.replace('&', '&amp;')
        txt = txt.replace('<', '&lt;')
        txt = txt.replace('>', '&gt;')
        txt = txt.replace('"', '&quot;')
        return txt


def makeExtension(*args, **kwargs):
    return RundocCodeExtension(*args, **kwargs)

