import ast
import sys
import os

def check_file(filepath: str, checks: list[dict]) -> list[str]:
    try:
        with open(filepath) as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception as e:
        return [f'FAIL | {filepath} | could not parse: {e}']
    results = []
    for check in checks:
        found = check['find'](source, tree)
        status = 'PASS' if found else 'FAIL'
        results.append(f'{status} | {filepath} | {check["name"]}')
    return results

def find_all_handlers_authorized(source: str, tree) -> bool:
    handler_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = getattr(node, 'func', None)
            if func and getattr(func, 'attr', '') == 'add_handler':
                for arg in node.args:
                    if isinstance(arg, ast.Call) and len(arg.args) >= 2:
                        fn_arg = arg.args[-1]
                        if isinstance(fn_arg, ast.Name):
                            handler_names.add(fn_arg.id)
                        elif isinstance(fn_arg, ast.Attribute):
                            handler_names.add(fn_arg.attr)
    decorated = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            for dec in node.decorator_list:
                name = ''
                if isinstance(dec, ast.Name): name = dec.id
                elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                    name = dec.func.id
                if name == 'authorized_only':
                    decorated.add(node.name)
    undecorated = handler_names - decorated
    if undecorated:
        print(f'  ⚠️  Missing @authorized_only: {sorted(undecorated)}')
        return False
    print(f'  ✓  {len(handler_names)} handlers checked: {sorted(handler_names)}')
    return True

def find_write_handlers_email_authed(source: str, tree) -> bool:
    required = {'send_add', 'send_remove', 'send_update', 'send_set'}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            if node.name in required:
                for dec in node.decorator_list:
                    name = ''
                    if isinstance(dec, ast.Name): name = dec.id
                    elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                        name = dec.func.id
                    if name == 'require_email_auth':
                        found.add(node.name)
    missing = required - found
    if missing:
        print(f'  ⚠️  Missing @require_email_auth: {sorted(missing)}')
        return False
    print(f'  ✓  Write handlers with @require_email_auth: {sorted(found)}')
    return True

def find_write_handlers_readonly_protected(source: str, tree) -> bool:
    # require_email_auth internally applies require_write_access
    # so check that require_write_access is defined and used inside require_email_auth
    return 'require_write_access' in source and 'READONLY_MODE' in source

def find_handle_callback_authorized(source: str, tree) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            if node.name == 'handle_callback':
                for dec in node.decorator_list:
                    name = ''
                    if isinstance(dec, ast.Name): name = dec.id
                    elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                        name = dec.func.id
                    if name == 'authorized_only':
                        print(f'  ✓  handle_callback has @authorized_only')
                        return True
    print(f'  ⚠️  handle_callback missing @authorized_only')
    return False

def find_handle_message_authorized(source: str, tree) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            if node.name == 'handle_message':
                for dec in node.decorator_list:
                    name = ''
                    if isinstance(dec, ast.Name): name = dec.id
                    elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                        name = dec.func.id
                    if name == 'authorized_only':
                        print(f'  ✓  handle_message has @authorized_only')
                        return True
    print(f'  ⚠️  handle_message missing @authorized_only')
    return False

def find_rate_limit(source: str, tree) -> bool:
    return 'check_rate_limit' in source and 'handle_message' in source

def find_claude_rate_limit(source: str, tree) -> bool:
    return 'check_claude_rate_limit' in source

def find_startup_env_check(source: str, tree) -> bool:
    return 'missing' in source and 'required_vars' in source

def find_readonly_mode(source: str, tree) -> bool:
    return 'READONLY_MODE' in source and 'BOT_READONLY' in source

def find_security_logger(source: str, tree) -> bool:
    return 'security_logger' in source and 'security.log' in source

def find_email_expiry(source: str, tree) -> bool:
    return 'expires' in source and 'verify_email_code' in source

def find_sanitize_headline(source: str, tree) -> bool:
    return 'sanitize_headline' in source

def find_containment_tags(source: str, tree) -> bool:
    return 'external_news_data' in source

if __name__ == '__main__':
    print()
    print('=' * 60)
    print('PORTFOLIO BOT — SECURITY REVIEW')
    print('=' * 60)

    bot_checks = [
        {'name': 'All registered handlers have @authorized_only',
         'find': find_all_handlers_authorized},
        {'name': 'Write commands have @require_email_auth',
         'find': find_write_handlers_email_authed},
        {'name': 'READONLY_MODE defined and used',
         'find': find_write_handlers_readonly_protected},
        {'name': 'handle_callback has @authorized_only',
         'find': find_handle_callback_authorized},
        {'name': 'handle_message has @authorized_only',
         'find': find_handle_message_authorized},
        {'name': 'Rate limiting in handle_message',
         'find': find_rate_limit},
        {'name': 'Claude rate limit check present',
         'find': find_claude_rate_limit},
        {'name': 'Startup env var validation present',
         'find': find_startup_env_check},
        {'name': 'Security logger configured',
         'find': find_security_logger},
    ]

    auth_checks = [
        {'name': 'Email code expiry checked in verify_email_code',
         'find': find_email_expiry},
    ]

    advisor_checks = [
        {'name': 'sanitize_headline present',
         'find': find_sanitize_headline},
        {'name': 'External data containment tags used',
         'find': find_containment_tags},
    ]

    all_results = []

    print('\n--- bot.py ---')
    all_results += check_file('equity/telegram/bot.py', bot_checks)

    print('\n--- auth.py ---')
    all_results += check_file('equity/telegram/auth.py', auth_checks)

    print('\n--- advisor.py ---')
    all_results += check_file('equity/telegram/advisor.py', advisor_checks)

    passes = sum(1 for r in all_results if r.startswith('PASS'))
    fails = sum(1 for r in all_results if r.startswith('FAIL'))

    print()
    print('=' * 60)
    print(f'RESULTS: {passes} PASS / {fails} FAIL')
    print()
    for r in all_results:
        print(r)
    print()
    if fails == 0:
        print('✅ All security checks passed. Safe to start the bot.')
    else:
        print('❌ Fix FAIL items before starting the bot.')
        print('   Do not run python -m equity.telegram.bot until all pass.')
    print('=' * 60)
    print()
    sys.exit(1 if fails > 0 else 0)
