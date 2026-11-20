// Ownership rejects duplicate clicks before React commits pending state.
export class PendingActions {
  private names = new Set<string>();
  begin(name: string): boolean {
    if (this.names.has(name)) return false;
    this.names.add(name);
    return true;
  }
  end(name: string) { this.names.delete(name); }
  has(name: string) { return this.names.has(name); }
  snapshot() { return new Set(this.names); }
}

export class LatestRequest {
  private generation = 0;
  begin() { return ++this.generation; }
  isLatest(generation: number) { return generation === this.generation; }
}

// Snapshots are JSON values; unchanged polls need not render again.
export function retainEqual<T>(current: T, next: T): T {
  return JSON.stringify(current) === JSON.stringify(next) ? current : next;
}
