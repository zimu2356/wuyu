"""Build a complete delivery index from confirmed successful save receipts."""
import hashlib
import json
from pathlib import Path
import sys

DATE='2026-10-12'

def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):result.update(block)
    return result.hexdigest()

def main():
    directory=Path(sys.argv[1]).resolve()
    receipt=json.loads(Path(sys.argv[2]).read_text())
    results=receipt['results']
    rows=[]
    for result in results:
        assert result['status']=='succeeded',result
        assert result.get('library_file_id') and result.get('local_metadata_applied') is True,result
        path=Path(result['local_path']).resolve()
        assert path.parent==directory and path.is_file() and not path.name.startswith('.')
        rows.append(dict(name=path.name,local_path=str(path),bytes=path.stat().st_size,
            sha256=digest(path),library_file_id=result['library_file_id'],file_id=result.get('file_id')))
    expected={p.name for p in directory.iterdir() if p.is_file() and not p.name.startswith('.')
        and p.name not in (f'WUYU_Relation_Downloads_{DATE}.md',f'WUYU_Relation_File_Index_{DATE}.json')}
    assert {r['name'] for r in rows}==expected
    by_name={r['name']:r for r in rows}
    def link(name):
        assert name in by_name
        return f"[{name}](sandbox:{directory/name})"
    main=[f'# 物欲 1.2：报告、完整源码和全部新原始记录\n',
        '实验冻结于 2026-10-12；2026-10-13 完成证据恢复和交付。',
        '关系读取的预设主比较通过，学习旧策略权重的额外收益仍未确立。默认 A 保持兼容；下一轮明确使用 R10，并保留 RT 对照。',
        '\n## 先看报告\n',link(f'WUYU_Relation_Report_{DATE}.html'),
        '独立 HTML，三张图已内嵌。包括机制、整体运行逻辑、主验证、辅助情境、长生命、干预诊断、限制和恢复来源。',
        link(f'WUYU_Relation_Report_{DATE}.md'),link(f'WUYU_Relation_HANDOFF_{DATE}.md'),
        '\n## 完整源码\n',link(f'WUYU_Relation_Source_{DATE}.zip'),
        'Source 单包包含物欲 1–8 与后续既有源码、当前完整主干、必要模型、测试、机制说明、冻结协议、摘要、分析和迁移示例。可单独运行，不需要先下载 Data。',
        '\n## 本轮全部原始记录\n',
        '将 Source 和以下所有 Data 包解压至同一父目录。共享 `WUYU_Whole_2026-10-12/` 顶层目录，成员不重叠。包含本轮全部开发/正式轨迹、诊断分支及相关存档；历史原始大数据继续使用之前交付的包，入口见 Source 中的历史下载说明。']
    table=['| 文件 | 大小 |','|---|---:|']
    for r in sorted(rows,key=lambda r:r['name']):
        if r['name'].startswith('WUYU_Relation_Data_'):
            table.append(f"| {link(r['name'])} | {r['bytes']/1e6:.1f} MB |")
    main.append('\n'.join(table))
    main.extend(['\n## 校验与图表\n',
        link(f'WUYU_Relation_Verification_{DATE}.json'),
        link(f'WUYU_Relation_SHA256SUMS_{DATE}.txt'),
        link(f'WUYU_Relation_Archive_Manifest_{DATE}.json'),
        '\n全量包成员哈希、ZIP CRC、无重叠覆盖和单独解压源码后的实际运行/存档续跑均通过。',
        '\n临时目录曾跨日重置，原 3,953 条摘要用于核对恢复；其余 431 条按原冻结方案重建。所有开发与正式生命重新审计，恢复不增加独立样本。详见报告与 `docs/RELATION_RECOVERY_2026-10-13.md`。',
        '\nSHA256SUMS 覆盖先完成的所有交付文件；本下载入口和 File_Index 在保存完成后生成，不递归纳入自身。'])
    for stem in ('Results','Scenarios','Timeline'):main.append(link(f'WUYU_Relation_{stem}_{DATE}.png'))
    (directory/f'WUYU_Relation_Downloads_{DATE}.md').write_text('\n\n'.join(main)+'\n')
    (directory/f'WUYU_Relation_File_Index_{DATE}.json').write_text(json.dumps(dict(
        complete=True,files=rows,first_batch_files=len(rows),total_bytes=sum(r['bytes'] for r in rows),
        note='Index and Downloads were generated after the first batch was saved; they are excluded from the first-batch self-checksums.'),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(indexed_files=len(rows),all_confirmed_saved=True)))

if __name__=='__main__':main()
