import json
import sys
from pathlib import Path


def check_result_file(result_path):

    print(f"\n{'='*70}")
    print(f"检查文件: {result_path.name}")
    print('='*70)
    
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        return False
    

    print("\n[检查1] 基本结构")
    has_tasks = "tasks" in result
    has_gateways = "gateways" in result
    has_dependencies = "dependencies" in result
    has_contexts = "contexts" in result
    
    print(f"  tasks: {'✅' if has_tasks else '❌'}")
    print(f"  gateways: {'✅' if has_gateways else '❌'}")
    print(f"  dependencies: {'✅' if has_dependencies else '❌'}")
    print(f"  contexts: {'✅' if has_contexts else '❌'}")
    
    if not (has_tasks and has_dependencies):
        print("❌ 缺少必要字段！")
        return False
    

    print("\n[检查2] 数据统计")
    tasks = result.get("tasks", [])
    gateways = result.get("gateways", [])
    dependencies = result.get("dependencies", [])
    contexts = result.get("contexts", [])
    
    print(f"  任务数量: {len(tasks)}")
    print(f"  网关数量: {len(gateways)}")
    print(f"  依赖数量: {len(dependencies)}")
    print(f"  上下文数量: {len(contexts)}")
    

    print("\n[检查3] Dependencies 质量")
    
    if len(dependencies) == 0:
        print("  ❌ 没有依赖关系！")
        return False
    

    has_gateway_deps = any(
        'gateway' in dep.get('source', '').lower() or 
        'gateway' in dep.get('target', '').lower() or
        'join' in dep.get('source', '').lower() or
        'join' in dep.get('target', '').lower()
        for dep in dependencies
    )
    
    if has_gateway_deps:
        print("  ✅ 包含网关相关的依赖")
    else:
        print("  ⚠️  可能只有顺序依赖（没有网关分支）")
    

    print(f"\n  前{min(5, len(dependencies))}条依赖:")
    for dep in dependencies[:5]:
        source = dep.get('source', 'unknown')
        target = dep.get('target', 'unknown')
        print(f"    {source} → {target}")
    

    print("\n[检查4] Contexts 质量")
    
    if len(contexts) == 0:
        print("  ⚠️  没有网关上下文（可能没有网关）")
    else:
        print("  ✅ 包含网关上下文")
        

        all_have_degrees = all(
            'in_degree' in ctx and 'out_degree' in ctx
            for ctx in contexts
        )
        
        if all_have_degrees:
            print("  ✅ 所有上下文都有入度/出度信息")
        else:
            print("  ❌ 部分上下文缺少入度/出度信息")
        

        print(f"\n  所有上下文:")
        for ctx in contexts:
            ctx_id = ctx.get('id', 'unknown')
            ctx_type = ctx.get('type', 'unknown')
            in_deg = ctx.get('in_degree', '?')
            out_deg = ctx.get('out_degree', '?')
            

            if in_deg == 1 and out_deg > 1:
                pattern = "fork (分支)"
            elif in_deg > 1 and out_deg == 1:
                pattern = "join (汇合)"
            else:
                pattern = "mixed"
            
            print(f"    {ctx_id} ({ctx_type}): in={in_deg}, out={out_deg} → {pattern}")
    

    print("\n[检查5] 完整性验证")
    

    all_ids = set()
    for task in tasks:
        all_ids.add(task.get('task_id', task.get('id', '')))
    for gw in gateways:
        all_ids.add(gw.get('id', ''))
    

    missing_ids = set()
    for dep in dependencies:
        source = dep.get('source', '')
        target = dep.get('target', '')
        

        if source and 'start' not in source.lower() and 'end' not in source.lower():
            if source not in all_ids and 'join' not in source.lower():
                missing_ids.add(source)
        
        if target and 'start' not in target.lower() and 'end' not in target.lower():
            if target not in all_ids and 'join' not in target.lower():
                missing_ids.add(target)
    
    if missing_ids:
        print(f"  ⚠️  发现 {len(missing_ids)} 个未定义的ID: {list(missing_ids)[:5]}")
    else:
        print("  ✅ 所有依赖中的ID都已定义")
    

    print("\n" + "="*70)
    
    critical_issues = []
    warnings = []
    
    if not has_dependencies:
        critical_issues.append("缺少 dependencies 字段")
    elif len(dependencies) == 0:
        critical_issues.append("dependencies 为空")
    
    if not has_contexts and len(gateways) > 0:
        warnings.append("有网关但没有 contexts")
    
    if not has_gateway_deps and len(gateways) > 0:
        warnings.append("有网关但 dependencies 中没有网关相关的依赖")
    
    if critical_issues:
        print("❌ 检查失败！")
        for issue in critical_issues:
            print(f"  - {issue}")
        return False
    
    if warnings:
        print("⚠️  检查通过但有警告")
        for warning in warnings:
            print(f"  - {warning}")
        return True
    
    print("✅ 检查完全通过！")
    return True


def main():
    """主函数"""
    print("="*70)
    print("网关修复验证工具")
    print("="*70)
    

    if len(sys.argv) > 1:

        result_files = [Path(sys.argv[1])]
    else:

        result_dir = Path("outputs")
        if not result_dir.exists():
            result_dir = Path(".")
        
        result_files = list(result_dir.glob("*.json"))
        
        if not result_files:
            print("❌ 未找到JSON结果文件")
            print(f"\n使用方法:")
            print(f"  python verify_gateway_fix.py [结果文件路径]")
            return
    

    results = {}
    for result_file in result_files:
        results[result_file.name] = check_result_file(result_file)
    

    print("\n" + "="*70)
    print("总结")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n检查了 {total} 个文件")
    print(f"  通过: {passed}")
    print(f"  失败: {total - passed}")
    
    if passed == total:
        print("\n✅ 所有文件检查通过！")
    elif passed > 0:
        print("\n⚠️  部分文件检查通过")
    else:
        print("\n❌ 所有文件检查失败")


if __name__ == "__main__":
    main()
