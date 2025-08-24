import os
import streamlit as st
import pandas as pd
import psycopg

# Basic page config
st.set_page_config(
    page_title="Literature OS",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Database connection
@st.cache_resource
def get_connection():
    return psycopg.connect(os.environ["DB_READER_DSN"])

# Data loading functions
@st.cache_data(ttl=300)
def get_papers(conn, search=None):
    query = """
        SELECT 
            id,
            title,
            journal,
            year,
            doi,
            citation_count
        FROM papers
        ORDER BY year DESC NULLS LAST, citation_count DESC NULLS LAST
        LIMIT 200
    """
    
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        
    return pd.DataFrame(
        rows,
        columns=["ID", "Title", "Journal", "Year", "DOI", "Citations"]
    )

# Main app
st.title("Literature OS")

try:
    # Connect to database
    conn = get_connection()
    
    # Load and display papers
    papers = get_papers(conn)
    
    if not papers.empty:
        st.dataframe(
            papers,
            use_container_width=True,
            hide_index=True
        )
        
        # Download option
        csv = papers.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download as CSV",
            csv,
            "papers.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.info("No papers found in the database.")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
import streamlit as st
import pandas as pd
import psycopg

# Choose a writable directory for Streamlit config (repo root works fine)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # .../literature-os
# If HOME is '/' or empty, redirect it to the project root
if os.environ.get("HOME", "/") in ("/", "", None):
    os.environ["HOME"] = str(PROJECT_ROOT)

# Ensure config dir exists and point XDG there too
os.environ.setdefault("XDG_CONFIG_HOME", os.environ["HOME"])
CONFIG_DIR = os.path.join(os.environ["HOME"], ".streamlit")
os.makedirs(CONFIG_DIR, exist_ok=True)
# --------------------------------------------------------------------


import streamlit as st
import pandas as pd
import psycopg

# Set up your page config
st.set_page_config(page_title="Literature OS", layout="wide")

st.caption("Last updated via GitHub → HF sync ✅")

# Initialize session state for error tracking
if 'error_log' not in st.session_state:
    st.session_state.error_log = []

def log_error(e: Exception, context: str):
    """Log error with context"""
    error_msg = f"{context}: {str(e)}"
    logger.error(error_msg)
    st.session_state.error_log.append(error_msg)
    return error_msg

@st.cache_resource
def conn():
    try:
        return psycopg.connect(
            os.environ["DB_READER_DSN"],
            application_name="literature-os-streamlit"
        )
    except Exception as e:
        log_error(e, "Database connection failed")
        raise

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_tags_years(_c):
    try:
        with _c.cursor() as cur:
            # Load tags with error handling
            try:
                cur.execute("SELECT DISTINCT tag FROM tags ORDER BY tag")
                tags = [r[0] for r in cur.fetchall() if r[0]]  # Filter out None/empty tags
            except Exception as e:
                log_error(e, "Failed to load tags")
                tags = []

            # Load years with error handling
            try:
                cur.execute("""
                    SELECT DISTINCT year 
                    FROM papers 
                    WHERE year IS NOT NULL 
                        AND year BETWEEN 1900 AND 2100
                    ORDER BY year DESC
                """)
                years = [r[0] for r in cur.fetchall() if r[0]]  # Filter out None years
            except Exception as e:
                log_error(e, "Failed to load years")
                years = list(range(2010, 2026))  # Fallback years

        return tags or [], years or list(range(2010, 2026))
    except Exception as e:
        log_error(e, "Failed to load metadata")
        return [], list(range(2010, 2026))

def safe_escape_like(text):
    """Safely escape LIKE pattern wildcards"""
    if not isinstance(text, str):
        return text
    return text.replace('%', '\\%').replace('_', '\\_')

def query(c, tags=None, y0=None, y1=None, q=None):
    """Query papers with safe parameter handling"""
    try:
        where = []; params = []
        
        # Handle tags (ensure it's a list and not empty)
        if tags and isinstance(tags, (list, tuple)) and len(tags) > 0:
            where.append("id IN (SELECT paper_id FROM tags WHERE tag = ANY(%s))")
            params.append(list(tags))  # Convert to list to be safe
            
        # Handle years (ensure they're valid numbers)
        if y0 is not None and y1 is not None:
            try:
                y0, y1 = int(y0), int(y1)
                where.append("(year IS NOT NULL AND year BETWEEN %s AND %s)")
                params += [y0, y1]
            except (ValueError, TypeError):
                st.warning("Invalid year range, ignoring year filter")
                
        # Handle search query (escape special characters)
        if q and isinstance(q, str) and q.strip():
            q = safe_escape_like(q.strip())
            where.append("(title ILIKE %s OR COALESCE(abstract, '') ILIKE %s)")
            params += [f"%{q}%", f"%{q}%"]
        
        # Build and execute query
        sql = """
            SELECT 
                id,
                COALESCE(title, '(No title)') as title,
                COALESCE(journal, '(No journal)') as journal,
                year,
                COALESCE(doi, '') as doi,
                COALESCE(citation_count, 0) as citation_count
            FROM papers
        """
        
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY year DESC NULLS LAST, citation_count DESC NULLS LAST LIMIT 200"
        
        # Debug information in expandable section
        with st.expander("🔍 Query Debug Info", expanded=False):
            st.code(sql)
            st.json({"parameters": str(params) if params else "None"})
        
        # Execute query with error handling
        with c.cursor() as cur:
            cur.execute(sql, params or None)
            rows = cur.fetchall()
            
        # Convert to DataFrame with proper types
        df = pd.DataFrame(rows, columns=["id","Title","Journal","Year","DOI","Citations"])
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Citations"] = pd.to_numeric(df["Citations"], errors="coerce").fillna(0).astype(int)
        return df
        
    except Exception as e:
        st.error(f"Query error: {str(e)}")
        st.code(f"SQL: {sql}\nParams: {params}")
        # Return empty DataFrame instead of raising
        return pd.DataFrame(columns=["id","Title","Journal","Year","DOI","Citations"])

# Initialize session state for filters
if 'filters' not in st.session_state:
    st.session_state.filters = {
        'search': '',
        'tags': [],
        'year_range': (2015, 2025)
    }

# Main app with comprehensive error handling
try:
    st.title("Literature OS - MVP")
    
    # Show any accumulated errors in debug section
    if st.session_state.error_log:
        with st.expander("🔍 Debug Log", expanded=False):
            for error in st.session_state.error_log:
                st.error(error)
    
    # Database connection with retry
    for attempt in range(3):  # Try 3 times
        try:
            c = conn()
            st.success("Connected to database ✅")
            break
        except Exception as e:
            if attempt == 2:  # Last attempt
                st.error("Database connection failed")
                st.error(str(e))
                st.stop()
            continue

    # Load metadata with fallbacks
    try:
        all_tags, years = load_tags_years(c)
    except Exception as e:
        st.error("Error loading filters")
        with st.expander("Error details"):
            st.code(str(e))
        # Use fallback values
        all_tags, years = [], list(range(2010, 2026))

    # Filter section in a container for visual grouping
    with st.container():
        st.subheader("📋 Filter Papers")
        
        try:
            # Search box with clear instructions
            search_query = st.text_input(
                "Search titles & abstracts",
                value=st.session_state.filters.get('search', ''),
                help="Enter keywords to search in paper titles and abstracts",
                key="search_input"
            )
            
            # Update session state
            st.session_state.filters['search'] = search_query

            # Year range with safe defaults
            col1, col2 = st.columns(2)
            with col1:
                try:
                    yr_min = min(y for y in years if isinstance(y, (int, float)))
                except ValueError:
                    yr_min = 2010
                try:
                    yr_max = max(y for y in years if isinstance(y, (int, float)))
                except ValueError:
                    yr_max = 2025
                
                year_range = st.slider(
                    "Year range",
                    min_value=yr_min,
                    max_value=yr_max,
                    value=st.session_state.filters.get('year_range', (max(2015, yr_min), yr_max)),
                    help="Filter papers by publication year",
                    key="year_slider"
                )
                
                # Update session state
                st.session_state.filters['year_range'] = year_range

            # Tags with categorized options
            with col2:
                prefix_map = {
                    "design:": "Research Design",
                    "domain:": "Domain",
                    "modality:": "Modality",
                    "dx:": "Diagnosis"
                }
                
                try:
                    # Group tags by prefix
                    tag_groups = {prefix: [] for prefix in prefix_map.keys()}
                    for tag in all_tags:
                        if not isinstance(tag, str):
                            continue
                        for prefix in prefix_map.keys():
                            if tag.startswith(prefix):
                                tag_groups[prefix].append(tag)
                                break
                    
                    # Create multiselect for each group that has tags
                    selected_tags = []
                    for prefix, label in prefix_map.items():
                        if tag_groups[prefix]:
                            group_key = f"tag_group_{prefix}"
                            group_tags = st.multiselect(
                                label,
                                options=sorted(tag_groups[prefix]),
                                default=st.session_state.filters.get('tags', []),
                                help=f"Select {label.lower()} tags to filter by",
                                key=group_key
                            )
                            selected_tags.extend(group_tags)
                    
                    # Update session state
                    st.session_state.filters['tags'] = selected_tags
                
                except Exception as e:
                    log_error(e, "Error in tag filtering")
                    selected_tags = []
                    
        except Exception as e:
            log_error(e, "Error in filter UI")
            search_query = ""
            year_range = (2015, 2025)
            selected_tags = []

    # Query and display results
    try:
        # Get filter values from session state
        yr0, yr1 = st.session_state.filters['year_range']
        df = query(
            c, 
            st.session_state.filters['tags'],
            yr0, yr1,
            st.session_state.filters['search']
        )
        
        # Results section
        st.subheader(f"📚 Results ({len(df)} papers)")
        
        if len(df) == 0:
            st.info("No papers found matching your criteria. Try adjusting the filters.")
        else:
            # Add clickable DOI links
            df['DOI'] = df['DOI'].apply(lambda x: f'[{x}](https://doi.org/{x})' if x else '')
            
            # Display the dataframe
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "DOI": st.column_config.LinkColumn("DOI"),
                    "Citations": st.column_config.NumberColumn("Citations", format="%d"),
                    "Year": st.column_config.NumberColumn("Year", format="%d")
                },
                hide_index=True
            )
            
            # Download button
            if not df.empty:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download as CSV",
                    csv,
                    "papers.csv",
                    "text/csv",
                    key='download-csv'
                )

    except Exception as e:
        st.error("Error loading results")
        with st.expander("Error details"):
            st.code(str(e))
        
except Exception as e:
    st.error("Application Error")
    with st.expander("Error details"):
        st.code(str(e))
