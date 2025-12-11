"""
Trade Storage Module
====================
Persistent storage for trades and equity curve using SQLite.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import json

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ClosedTrade:
    """Record of a closed trade."""
    id: Optional[int] = None
    timestamp: str = ""
    symbol: str = ""
    side: str = ""  # 'buy' or 'sell' (entry side)
    qty: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl_abs: float = 0.0
    pnl_pct: float = 0.0
    equity_after_trade: float = 0.0
    hold_duration_seconds: int = 0
    entry_order_id: str = ""
    exit_order_id: str = ""
    notes: str = ""


@dataclass
class EquitySnapshot:
    """Point-in-time equity snapshot."""
    id: Optional[int] = None
    timestamp: str = ""
    equity: float = 0.0
    cash: float = 0.0
    positions_value: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0


class TradeStore:
    """
    SQLite-backed trade storage.

    Stores:
    - Closed trades with full details
    - Equity curve snapshots
    - Open position tracking
    """

    def __init__(self, db_path: Path = Path('data/trades.db')):
        """
        Initialize trade store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        logger.info(f"Trade store initialized: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Closed trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS closed_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                pnl_abs REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                equity_after_trade REAL NOT NULL,
                hold_duration_seconds INTEGER DEFAULT 0,
                entry_order_id TEXT,
                exit_order_id TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Equity curve table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS equity_curve (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                daily_pnl_pct REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Open positions table (for tracking)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS open_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                entry_order_id TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON closed_trades(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON closed_trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_curve(timestamp)')

        conn.commit()
        conn.close()

    def record_trade(self, trade: ClosedTrade) -> int:
        """
        Record a closed trade.

        Args:
            trade: ClosedTrade object

        Returns:
            Trade ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO closed_trades
            (timestamp, symbol, side, qty, entry_price, exit_price,
             pnl_abs, pnl_pct, equity_after_trade, hold_duration_seconds,
             entry_order_id, exit_order_id, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.timestamp or datetime.now().isoformat(),
            trade.symbol,
            trade.side,
            trade.qty,
            trade.entry_price,
            trade.exit_price,
            trade.pnl_abs,
            trade.pnl_pct,
            trade.equity_after_trade,
            trade.hold_duration_seconds,
            trade.entry_order_id,
            trade.exit_order_id,
            trade.notes,
        ))

        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"Recorded trade #{trade_id}: {trade.symbol} PnL ${trade.pnl_abs:,.2f}")

        return trade_id

    def record_equity_snapshot(self, snapshot: EquitySnapshot) -> int:
        """
        Record an equity snapshot.

        Args:
            snapshot: EquitySnapshot object

        Returns:
            Snapshot ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO equity_curve
            (timestamp, equity, cash, positions_value, daily_pnl, daily_pnl_pct)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            snapshot.timestamp or datetime.now().isoformat(),
            snapshot.equity,
            snapshot.cash,
            snapshot.positions_value,
            snapshot.daily_pnl,
            snapshot.daily_pnl_pct,
        ))

        snapshot_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return snapshot_id

    def track_position_open(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        entry_order_id: str = "",
    ) -> None:
        """Track a newly opened position."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO open_positions
            (symbol, side, qty, entry_price, entry_time, entry_order_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            side,
            qty,
            entry_price,
            datetime.now().isoformat(),
            entry_order_id,
            datetime.now().isoformat(),
        ))

        conn.commit()
        conn.close()

    def track_position_close(
        self,
        symbol: str,
        exit_price: float,
        equity_after: float,
        exit_order_id: str = "",
    ) -> Optional[ClosedTrade]:
        """
        Track position close and record trade.

        Returns:
            ClosedTrade if position was found, None otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get open position
        cursor.execute('SELECT * FROM open_positions WHERE symbol = ?', (symbol,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return None

        # Calculate trade details
        entry_time = datetime.fromisoformat(row['entry_time'])
        hold_duration = int((datetime.now() - entry_time).total_seconds())

        qty = row['qty']
        entry_price = row['entry_price']
        side = row['side']

        # Calculate PnL
        if side == 'long':
            pnl_abs = (exit_price - entry_price) * qty
        else:
            pnl_abs = (entry_price - exit_price) * qty

        pnl_pct = (pnl_abs / (entry_price * qty)) * 100 if entry_price > 0 else 0

        # Create trade record
        trade = ClosedTrade(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_abs=pnl_abs,
            pnl_pct=pnl_pct,
            equity_after_trade=equity_after,
            hold_duration_seconds=hold_duration,
            entry_order_id=row['entry_order_id'] or "",
            exit_order_id=exit_order_id,
        )

        # Record trade
        trade.id = self.record_trade(trade)

        # Remove from open positions
        cursor.execute('DELETE FROM open_positions WHERE symbol = ?', (symbol,))
        conn.commit()
        conn.close()

        return trade

    def get_trades(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        symbol: Optional[str] = None,
        limit: int = 1000,
    ) -> List[ClosedTrade]:
        """
        Get closed trades with optional filters.

        Args:
            since: Start datetime filter
            until: End datetime filter
            symbol: Symbol filter
            limit: Maximum number of trades to return

        Returns:
            List of ClosedTrade objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = 'SELECT * FROM closed_trades WHERE 1=1'
        params = []

        if since:
            query += ' AND timestamp >= ?'
            params.append(since.isoformat())

        if until:
            query += ' AND timestamp <= ?'
            params.append(until.isoformat())

        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)

        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        trades = []
        for row in rows:
            trades.append(ClosedTrade(
                id=row['id'],
                timestamp=row['timestamp'],
                symbol=row['symbol'],
                side=row['side'],
                qty=row['qty'],
                entry_price=row['entry_price'],
                exit_price=row['exit_price'],
                pnl_abs=row['pnl_abs'],
                pnl_pct=row['pnl_pct'],
                equity_after_trade=row['equity_after_trade'],
                hold_duration_seconds=row['hold_duration_seconds'],
                entry_order_id=row['entry_order_id'] or "",
                exit_order_id=row['exit_order_id'] or "",
                notes=row['notes'] or "",
            ))

        return trades

    def get_equity_curve(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 10000,
    ) -> List[EquitySnapshot]:
        """Get equity curve snapshots."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = 'SELECT * FROM equity_curve WHERE 1=1'
        params = []

        if since:
            query += ' AND timestamp >= ?'
            params.append(since.isoformat())

        if until:
            query += ' AND timestamp <= ?'
            params.append(until.isoformat())

        query += ' ORDER BY timestamp ASC LIMIT ?'
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        snapshots = []
        for row in rows:
            snapshots.append(EquitySnapshot(
                id=row['id'],
                timestamp=row['timestamp'],
                equity=row['equity'],
                cash=row['cash'],
                positions_value=row['positions_value'],
                daily_pnl=row['daily_pnl'],
                daily_pnl_pct=row['daily_pnl_pct'],
            ))

        return snapshots

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all tracked open positions."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM open_positions ORDER BY entry_time DESC')
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_trade_stats(self) -> Dict[str, Any]:
        """Get aggregate trade statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl_abs > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl_abs < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl_abs) as total_pnl,
                AVG(pnl_abs) as avg_pnl,
                AVG(CASE WHEN pnl_abs > 0 THEN pnl_abs END) as avg_win,
                AVG(CASE WHEN pnl_abs < 0 THEN pnl_abs END) as avg_loss,
                MAX(pnl_abs) as max_win,
                MIN(pnl_abs) as max_loss,
                SUM(CASE WHEN pnl_abs > 0 THEN pnl_abs ELSE 0 END) as gross_profit,
                SUM(CASE WHEN pnl_abs < 0 THEN ABS(pnl_abs) ELSE 0 END) as gross_loss
            FROM closed_trades
        ''')

        row = cursor.fetchone()
        conn.close()

        if not row or row['total_trades'] == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_win': 0.0,
                'max_loss': 0.0,
                'profit_factor': 0.0,
            }

        total = row['total_trades']
        wins = row['winning_trades'] or 0
        gross_profit = row['gross_profit'] or 0
        gross_loss = row['gross_loss'] or 0

        return {
            'total_trades': total,
            'winning_trades': wins,
            'losing_trades': row['losing_trades'] or 0,
            'win_rate': (wins / total * 100) if total > 0 else 0,
            'total_pnl': row['total_pnl'] or 0,
            'avg_pnl': row['avg_pnl'] or 0,
            'avg_win': row['avg_win'] or 0,
            'avg_loss': row['avg_loss'] or 0,
            'max_win': row['max_win'] or 0,
            'max_loss': row['max_loss'] or 0,
            'profit_factor': (gross_profit / gross_loss) if gross_loss > 0 else float('inf'),
        }

    def export_to_csv(self, output_path: Path) -> None:
        """Export trades to CSV file."""
        import csv

        trades = self.get_trades(limit=100000)

        with open(output_path, 'w', newline='') as f:
            if not trades:
                f.write("No trades to export\n")
                return

            writer = csv.DictWriter(f, fieldnames=list(asdict(trades[0]).keys()))
            writer.writeheader()

            for trade in trades:
                writer.writerow(asdict(trade))

        logger.info(f"Exported {len(trades)} trades to {output_path}")

    def reconcile_positions(
        self,
        alpaca_positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Reconcile Alpaca positions with SQLite open_positions table.

        Args:
            alpaca_positions: List of position dicts from Alpaca
                Each dict should have: symbol, qty, avg_entry_price, market_value

        Returns:
            Dict with reconciliation results:
                - matched: positions that match
                - missing_in_db: positions in Alpaca but not in DB
                - missing_in_alpaca: positions in DB but not in Alpaca
                - qty_mismatch: positions with quantity differences
                - needs_sync: True if any discrepancies found
        """
        # Get DB positions
        db_positions = {p['symbol']: p for p in self.get_open_positions()}

        # Build Alpaca positions dict
        alpaca_pos_dict = {}
        for pos in alpaca_positions:
            alpaca_pos_dict[pos['symbol']] = {
                'qty': float(pos['qty']),
                'avg_entry_price': float(pos.get('avg_entry_price', 0)),
                'market_value': float(pos.get('market_value', 0)),
            }

        # Find discrepancies
        all_symbols = set(db_positions.keys()) | set(alpaca_pos_dict.keys())

        matched = []
        missing_in_db = []
        missing_in_alpaca = []
        qty_mismatch = []

        for symbol in all_symbols:
            in_db = symbol in db_positions
            in_alpaca = symbol in alpaca_pos_dict

            if in_db and in_alpaca:
                db_qty = db_positions[symbol]['qty']
                alpaca_qty = alpaca_pos_dict[symbol]['qty']

                if abs(db_qty - alpaca_qty) < 0.01:  # Allow small float diff
                    matched.append(symbol)
                else:
                    qty_mismatch.append({
                        'symbol': symbol,
                        'db_qty': db_qty,
                        'alpaca_qty': alpaca_qty,
                        'difference': alpaca_qty - db_qty,
                    })
            elif in_alpaca and not in_db:
                missing_in_db.append({
                    'symbol': symbol,
                    'qty': alpaca_pos_dict[symbol]['qty'],
                    'avg_entry_price': alpaca_pos_dict[symbol]['avg_entry_price'],
                })
            elif in_db and not in_alpaca:
                missing_in_alpaca.append({
                    'symbol': symbol,
                    'qty': db_positions[symbol]['qty'],
                    'entry_price': db_positions[symbol]['entry_price'],
                })

        needs_sync = bool(missing_in_db or missing_in_alpaca or qty_mismatch)

        result = {
            'matched': matched,
            'missing_in_db': missing_in_db,
            'missing_in_alpaca': missing_in_alpaca,
            'qty_mismatch': qty_mismatch,
            'needs_sync': needs_sync,
            'total_alpaca': len(alpaca_pos_dict),
            'total_db': len(db_positions),
        }

        if needs_sync:
            logger.warning(f"Position reconciliation found discrepancies: "
                          f"missing_in_db={len(missing_in_db)}, "
                          f"missing_in_alpaca={len(missing_in_alpaca)}, "
                          f"qty_mismatch={len(qty_mismatch)}")
        else:
            logger.info(f"Position reconciliation OK: {len(matched)} positions matched")

        return result

    def sync_from_alpaca(
        self,
        alpaca_positions: List[Dict[str, Any]],
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Sync open_positions table from Alpaca (source of truth).

        Args:
            alpaca_positions: List of position dicts from Alpaca
            force: If True, overwrite DB even if positions exist

        Returns:
            Dict with sync results
        """
        reconciliation = self.reconcile_positions(alpaca_positions)

        if not reconciliation['needs_sync'] and not force:
            return {'action': 'none', 'reason': 'already_synced'}

        conn = self._get_connection()
        cursor = conn.cursor()

        added = []
        removed = []
        updated = []

        # Add positions missing in DB
        for pos in reconciliation['missing_in_db']:
            cursor.execute('''
                INSERT OR REPLACE INTO open_positions
                (symbol, side, qty, entry_price, entry_time, entry_order_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                pos['symbol'],
                'long',  # Assume long for now
                pos['qty'],
                pos['avg_entry_price'],
                datetime.now().isoformat(),
                'synced_from_alpaca',
                datetime.now().isoformat(),
            ))
            added.append(pos['symbol'])
            logger.info(f"Added missing position: {pos['symbol']} qty={pos['qty']}")

        # Remove positions not in Alpaca (likely sold externally)
        for pos in reconciliation['missing_in_alpaca']:
            cursor.execute('DELETE FROM open_positions WHERE symbol = ?', (pos['symbol'],))
            removed.append(pos['symbol'])
            logger.info(f"Removed stale position: {pos['symbol']}")

        # Update qty mismatches
        for pos in reconciliation['qty_mismatch']:
            cursor.execute('''
                UPDATE open_positions
                SET qty = ?, updated_at = ?
                WHERE symbol = ?
            ''', (pos['alpaca_qty'], datetime.now().isoformat(), pos['symbol']))
            updated.append(pos['symbol'])
            logger.info(f"Updated position qty: {pos['symbol']} {pos['db_qty']} -> {pos['alpaca_qty']}")

        conn.commit()
        conn.close()

        return {
            'action': 'synced',
            'added': added,
            'removed': removed,
            'updated': updated,
        }

    def clear_open_positions(self) -> int:
        """Clear all open positions (for reset/testing)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM open_positions')
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        logger.warning(f"Cleared {deleted} open positions from database")
        return deleted


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    # Test trade store
    with tempfile.TemporaryDirectory() as tmpdir:
        store = TradeStore(Path(tmpdir) / 'test.db')

        # Record some test trades
        store.record_trade(ClosedTrade(
            symbol='AAPL',
            side='long',
            qty=10,
            entry_price=150.00,
            exit_price=155.00,
            pnl_abs=50.00,
            pnl_pct=3.33,
            equity_after_trade=100050.00,
        ))

        store.record_trade(ClosedTrade(
            symbol='MSFT',
            side='long',
            qty=5,
            entry_price=300.00,
            exit_price=290.00,
            pnl_abs=-50.00,
            pnl_pct=-3.33,
            equity_after_trade=100000.00,
        ))

        # Get stats
        stats = store.get_trade_stats()
        print("Trade Stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # Get trades
        trades = store.get_trades()
        print(f"\nRecorded {len(trades)} trades")
