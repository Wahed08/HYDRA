def detect_missing_malloc_check(code: str) -> int:
    alloc_pattern = r'\b(\w+)\s*\*\s*(\w+)\s*=\s*\([^\)]*\)\s*(malloc|calloc|realloc)\s*\([^;]+?\);|\b(\w+)\s*\*\s*(\w+)\s*=\s*(malloc|calloc|realloc)\s*\([^;]+?\);'
    matches = re.findall(alloc_pattern, code)
    ptrs = [match[1] if match[1] else match[4] for match in matches if match[1] or match[4]]
    for ptr in ptrs:
        is_checked = False
        is_used = False
        null_check_patterns = [rf'if\s*\(\s*{ptr}\s*\)', rf'if\s*\(\s*{ptr}\s*!=\s*NULL\s*\)', rf'if\s*\(\s*!{ptr}\s*\)', rf'if\s*\(\s*NULL\s*!=\s*{ptr}\s*\)']
        usage_patterns = [rf'{ptr}\s*\[', rf'\*\s*{ptr}', rf'{ptr}\s*->', rf'{ptr}\s*\(', rf'{ptr}\s*=']
        for pat in null_check_patterns:
            if re.search(pat, code):
                is_checked = True
                break
        for pat in usage_patterns:
            if re.search(pat, code):
                is_used = True
                break
        if is_used and not is_checked:
            return 1
    return 0

def missing_null_check_func(code):
    functions = re.findall(r'\b\w+\s+\w+\s*\([^)]*\)\s*\{[^{}]*\}', code, re.DOTALL)
    for func in functions:
        ptr_decl = re.findall(r'\b\w+\s*\*\s*(\w+)\s*=\s*[^;]+;', func)
        ptr_func_decl = re.findall(r'\b\w+\s*\*\s*(\w+)\s*;', func)
        ptr_vars = list(set(ptr_decl + ptr_func_decl))
        for ptr in ptr_vars:
            unsafe_use = re.search(rf'[^a-zA-Z0-9_]({ptr})(->|\[\d*]|[^\w]?\*)|\b{ptr}\s*\(', func)
            if not unsafe_use:
                continue
            null_check = re.search(rf'if\s*\(\s*(!\s*{ptr}|{ptr}\s*==\s*NULL|{ptr}\s*!=\s*NULL)\s*\)', func)
            if not null_check:
                return 1
    return 0

def detect_race_condition(code: str) -> bool:
    field_assignment_pattern = r'\b\w+\s*(->|\.)\s*\w+\s*=.*?;'
    control_block_pattern = r'\b(if|while|for|switch)\s*\([^\)]+\)\s*{[^}]*' + field_assignment_pattern + r'[^}]*}'
    matches_control_blocks = re.findall(control_block_pattern, code, re.DOTALL)
    locking_primitive_pattern = r'\b(mutex_lock|pthread_mutex_lock|spin_lock)\b'
    unlocking_primitive_pattern = r'\b(mutex_unlock|pthread_mutex_unlock|spin_unlock)\b'
    has_locking = re.search(locking_primitive_pattern, code)
    has_unlocking = re.search(unlocking_primitive_pattern, code)
    if matches_control_blocks and not (has_locking and has_unlocking):
        return True
    return False

def logging_but_no_blocking(code):
    log_lines = re.findall(r'(syslog\([^)]*\)|printk\([^)]*\))', code)
    for line in log_lines:
        log_index = code.find(line)
        snippet = code[log_index:log_index+150]
        if not re.search(r'(return|exit|break|continue)', snippet):
            return 1
    return 0

def split_into_functions(code):
    functions = re.findall(r'([a-zA-Z_][a-zA-Z0-9_\*\s]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*?\)\s*\{.*?\})', code, re.DOTALL)
    return functions

def missing_bounds_check(code):
    functions = split_into_functions(code)
    risky_keywords = ['recv', 'read', 'strcpy', 'memcpy', 'gets', 'strcat', 'write']
    for func in functions:
        found_risky = any(kw in func for kw in risky_keywords)
        has_check = any('if' in line and ('<' in line or '>' in line or '<=' in line or '>=' in line) for line in func.splitlines())
        if found_risky and not has_check:
            return 1
    return 0

def analyze_risks(code):
    risk_flags = {
        "Missing Null Check": missing_null_check_func(code),
        "Race Condition": detect_race_condition(code),
        "Missing Bounds Check": missing_bounds_check(code),
        "Unsafe Memory Allocation": detect_missing_malloc_check(code),
        "Logging Without Halting": logging_but_no_blocking(code),
        "issue_detected": 0
    }
    if any(v != 0 for v in risk_flags.values() if v is not False):
        risk_flags["issue_detected"] = 1
    return risk_flags