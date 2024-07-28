"""Fix TrafficSignal

Revision ID: a526efba7221
Revises: b84c9a597954
Create Date: 2024-07-28 19:42:25.678328

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a526efba7221'
down_revision: Union[str, None] = 'b84c9a597954'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('traffic_signals', 'time_from_last_change')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('traffic_signals', sa.Column('time_from_last_change', sa.INTEGER(), autoincrement=False, nullable=True))
    # ### end Alembic commands ###
