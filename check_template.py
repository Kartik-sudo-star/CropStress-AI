with open('frontend/src/pages/Processing.jsx', encoding='utf-8') as f:
    t = f.read()

import re
matches = list(re.finditer(r'`[^`]*`', t))
print(f'Template literals: {len(matches)}')
for m in matches:
    content = m.group()
    if '/' in content and not content.startswith('`//') and not content.startswith('` '):
        line = t[:m.start()].count('\n') + 1
        print(f'Line {line}: {content[:120]}')